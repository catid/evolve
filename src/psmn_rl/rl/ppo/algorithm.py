from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from psmn_rl.config import ExperimentConfig
from psmn_rl.envs.registry import make_eval_env, make_reset_seeds, make_vector_env
from psmn_rl.logging import LOGGER, RunLogger
from psmn_rl.metrics import MetricAggregator, scalarize_metrics
from psmn_rl.models.common import ActorCriticModel
from psmn_rl.rl.distributed.ddp import (
    DistributedContext,
    barrier,
    broadcast_scalar_dict,
    reduce_scalar_dict,
    unwrap_ddp,
)
from psmn_rl.rl.rollout.storage import RolloutStorage
from psmn_rl.utils.io import get_git_commit, save_json, try_get_gpu_utilization
from psmn_rl.utils.seed import capture_rng_state, restore_rng_state, set_seed


@dataclass
class TrainResult:
    output_dir: str
    latest_checkpoint: str
    final_metrics: dict[str, float]


def prepare_obs(obs: dict[str, np.ndarray] | np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    if isinstance(obs, dict):
        filtered: dict[str, torch.Tensor] = {}
        for key in ("image", "direction", "pixels"):
            if key in obs:
                filtered[key] = torch.as_tensor(obs[key], device=device)
        return filtered
    return {"pixels": torch.as_tensor(obs, device=device)}


def prepare_done(done: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(done, device=device, dtype=torch.bool)


def _state_index(state: dict[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[index] for key, value in state.items()}


def _stack_raw_obs(obs_list: list[dict[str, Any] | np.ndarray]) -> dict[str, np.ndarray] | np.ndarray:
    first = obs_list[0]
    if isinstance(first, dict):
        return {
            key: np.stack([np.asarray(obs[key]) for obs in obs_list], axis=0)
            for key in ("image", "direction", "pixels")
            if key in first
        }
    return np.stack([np.asarray(obs) for obs in obs_list], axis=0)


def _extract_final_obs_batch(
    info: dict[str, Any],
    mask: np.ndarray,
) -> tuple[torch.Tensor, list[dict[str, Any] | np.ndarray]] | None:
    final_obs = info.get("final_obs")
    final_mask = info.get("_final_obs")
    if final_obs is None or final_mask is None:
        return None
    present = np.asarray(final_mask, dtype=bool)
    selected = np.nonzero(mask & present)[0]
    if selected.size == 0:
        return None
    obs_list = [final_obs[index] for index in selected]
    return torch.as_tensor(selected, dtype=torch.long), obs_list


def _episode_successes(
    reward: np.ndarray,
    done: np.ndarray,
    info: dict[str, Any],
) -> np.ndarray:
    success = np.zeros_like(reward, dtype=np.float32)
    if not np.any(done):
        return success
    final_info = info.get("final_info")
    final_info_mask = np.asarray(info.get("_final_info", np.zeros_like(done, dtype=bool)), dtype=bool)
    for index, finished in enumerate(done):
        if not finished:
            continue
        if isinstance(final_info, dict) and final_info_mask[index]:
            env_info = {key: value[index] for key, value in final_info.items() if len(value) > index}
            for key in ("success", "is_success", "goal_reached", "completed"):
                if key in env_info:
                    success[index] = float(env_info[key])
                    break
            else:
                success[index] = float(reward[index] > 0.0)
        else:
            success[index] = float(reward[index] > 0.0)
    return success


def _apply_truncation_bootstrap(
    model: nn.Module,
    next_state: dict[str, torch.Tensor],
    truncated: np.ndarray,
    info: dict[str, Any],
    reward_t: torch.Tensor,
    ctx: DistributedContext,
    gamma: float,
) -> None:
    final_batch = _extract_final_obs_batch(info, truncated)
    if final_batch is None:
        return
    selected, obs_list = final_batch
    final_obs_t = prepare_obs(_stack_raw_obs(obs_list), ctx.device)
    final_state = _state_index(next_state, selected.to(device=ctx.device))
    bootstrap_done = torch.zeros(selected.numel(), device=ctx.device, dtype=torch.bool)
    with torch.no_grad():
        with _autocast_context(ctx):
            final_value = model.forward(final_obs_t, state=final_state, done=bootstrap_done).value.detach()
    reward_t[selected.to(device=ctx.device)] += gamma * final_value


def _autocast_context(ctx: DistributedContext):
    if ctx.autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=ctx.device.type, dtype=ctx.autocast_dtype)


def _checkpoint_path(config: ExperimentConfig, name: str) -> str:
    return str(Path(config.logging.output_dir) / name)


def policy_act(
    model: nn.Module,
    obs: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    done: torch.Tensor,
    greedy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor | float], dict[str, torch.Tensor]]:
    output = model(obs, state=state, done=done)
    dist = unwrap_ddp(model).get_dist(output.logits)
    action = output.logits.argmax(dim=-1) if greedy else dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob, output.value, output.next_state, output.metrics, output.aux_losses


def policy_evaluate_actions(
    model: nn.Module,
    obs: dict[str, torch.Tensor],
    actions: torch.Tensor,
    state: dict[str, torch.Tensor],
    done: torch.Tensor,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor | float] | dict[str, torch.Tensor]]:
    output = model(obs, state=state, done=done)
    dist = unwrap_ddp(model).get_dist(output.logits)
    return {
        "log_prob": dist.log_prob(actions),
        "entropy": dist.entropy(),
        "value": output.value,
        "metrics": output.metrics,
        "next_state": output.next_state,
        "aux_losses": output.aux_losses,
    }


def save_checkpoint(
    config: ExperimentConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    update: int,
    step: int,
    is_main_process: bool,
) -> str | None:
    if not is_main_process or not config.logging.save_checkpoints:
        return None
    output_dir = Path(config.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": unwrap_ddp(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "step": step,
        "config": config.to_dict(),
        "git_commit": get_git_commit(),
        "rng_state": capture_rng_state(),
    }
    latest_path = output_dir / "latest.pt"
    torch.save(checkpoint, latest_path)
    return str(latest_path)


def evaluate_policy(
    config: ExperimentConfig,
    model: ActorCriticModel,
    ctx: DistributedContext,
    episodes: int | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    try:
        if ctx.is_main_process:
            rng_state = capture_rng_state()
            set_seed(config.seed + 999_999, deterministic=config.system.deterministic)
            eval_env = make_eval_env(config.env, seed=config.seed + 999, world_rank=0)
            eval_seed = make_reset_seeds(config.env.num_eval_envs, config.seed + 999, world_rank=0)
            obs, _ = eval_env.reset(seed=eval_seed)
            obs_t = prepare_obs(obs, ctx.device)
            done_t = torch.ones(config.env.num_eval_envs, device=ctx.device, dtype=torch.bool)
            state = model.initial_state(config.env.num_eval_envs, ctx.device)
            returns = np.zeros(config.env.num_eval_envs, dtype=np.float32)
            lengths = np.zeros(config.env.num_eval_envs, dtype=np.int32)
            finished_returns: list[float] = []
            finished_successes: list[float] = []
            finished_lengths: list[float] = []
            target_episodes = episodes or config.evaluation.episodes
            try:
                while len(finished_returns) < target_episodes:
                    with torch.inference_mode():
                        action, _, _, state, _, _ = policy_act(
                            model,
                            obs_t,
                            state=state,
                            done=done_t,
                            greedy=config.evaluation.greedy,
                        )
                    next_obs, reward, terminated, truncated, info = eval_env.step(action.cpu().numpy())
                    done_np = np.logical_or(terminated, truncated)
                    step_success = _episode_successes(reward, done_np, info)
                    returns += reward
                    lengths += 1
                    for index, finished in enumerate(done_np):
                        if finished:
                            finished_returns.append(float(returns[index]))
                            finished_successes.append(float(step_success[index]))
                            finished_lengths.append(float(lengths[index]))
                            returns[index] = 0.0
                            lengths[index] = 0
                    obs_t = prepare_obs(next_obs, ctx.device)
                    done_t = prepare_done(done_np, ctx.device)
            finally:
                eval_env.close()
                restore_rng_state(rng_state)
            metrics = {
                "eval_return": float(np.mean(finished_returns[:target_episodes])),
                "eval_success_rate": float(np.mean(finished_successes[:target_episodes])),
                "eval_episode_length": float(np.mean(finished_lengths[:target_episodes])) if finished_lengths else 0.0,
            }
        else:
            metrics = {}
    finally:
        if was_training:
            model.train()
    return broadcast_scalar_dict(metrics, ctx)


def train(
    config: ExperimentConfig,
    model: ActorCriticModel,
    optimizer: torch.optim.Optimizer,
    ctx: DistributedContext,
    max_updates: int | None = None,
    start_update: int = 0,
    start_step: int = 0,
) -> TrainResult:
    logger = RunLogger(config, enabled=ctx.is_main_process)
    envs = make_vector_env(config.env, seed=config.seed, world_rank=ctx.rank)
    train_seed = make_reset_seeds(config.env.num_envs, config.seed, world_rank=ctx.rank)
    obs, _ = envs.reset(seed=train_seed)
    obs_t = prepare_obs(obs, ctx.device)
    done_t = torch.ones(config.env.num_envs, device=ctx.device, dtype=torch.bool)
    state = unwrap_ddp(model).initial_state(config.env.num_envs, ctx.device)

    total_updates = max_updates or config.ppo.total_updates
    global_step = start_step
    start_time = time.perf_counter()
    episode_returns = np.zeros(config.env.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(config.env.num_envs, dtype=np.int32)
    latest_checkpoint = ""

    metadata = {
        "git_commit": get_git_commit(),
        "rank": ctx.rank,
        "world_size": ctx.world_size,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "seed": config.seed,
    }
    if ctx.is_main_process:
        save_json(f"{config.logging.output_dir}/run_meta.json", metadata)

    update_metrics = {
        "global_step": float(global_step),
        "wall_clock_seconds": 0.0,
        "throughput_fps": 0.0,
        "train/episode_return": 0.0,
        "train/success_rate": 0.0,
        "train/episode_length": 0.0,
        "train/episodes_completed": 0.0,
        "gpu_utilization": 0.0,
    }

    for update in range(start_update + 1, total_updates + 1):
        if config.ppo.anneal_lr:
            frac = 1.0 - (update - 1) / max(total_updates, 1)
            optimizer.param_groups[0]["lr"] = frac * config.ppo.learning_rate

        rollout = RolloutStorage()
        rollout_metrics = MetricAggregator()
        update_completed_returns: list[float] = []
        update_completed_successes: list[float] = []
        update_completed_lengths: list[float] = []

        for _ in range(config.ppo.rollout_steps):
            with torch.no_grad():
                with _autocast_context(ctx):
                    action, log_prob, value, next_state, metrics, _ = policy_act(
                        model,
                        obs_t,
                        state=state,
                        done=done_t,
                    )

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            reward_t = torch.as_tensor(reward, device=ctx.device, dtype=torch.float32)
            truncated_np = np.asarray(truncated, dtype=bool)
            if truncated_np.any():
                _apply_truncation_bootstrap(
                    model=model,
                    next_state=next_state,
                    truncated=truncated_np,
                    info=info,
                    reward_t=reward_t,
                    ctx=ctx,
                    gamma=config.ppo.gamma,
                )
            next_done_np = np.logical_or(terminated, truncated)
            next_done_t = prepare_done(next_done_np, ctx.device)
            rollout.add(obs_t, state, done_t, action, log_prob, value, reward_t, next_done_t)
            rollout_metrics.update(metrics)

            episode_returns += reward
            episode_lengths += 1
            step_success = _episode_successes(reward, next_done_np, info)
            for env_index, finished in enumerate(next_done_np):
                if finished:
                    update_completed_returns.append(float(episode_returns[env_index]))
                    update_completed_successes.append(float(step_success[env_index]))
                    update_completed_lengths.append(float(episode_lengths[env_index]))
                    episode_returns[env_index] = 0.0
                    episode_lengths[env_index] = 0

            obs_t = prepare_obs(next_obs, ctx.device)
            done_t = next_done_t
            state = next_state
            global_step += config.env.num_envs * ctx.world_size

        with torch.no_grad():
            with _autocast_context(ctx):
                bootstrap = model.forward(obs_t, state=state, done=done_t)
        batch = rollout.compute_returns_and_advantages(
            last_value=bootstrap.value.detach(),
            last_done=done_t,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
        ).flatten()
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        batch.advantages = advantages
        returns_var = torch.var(batch.returns, unbiased=False)
        explained_variance = 0.0
        if returns_var.item() > 1e-8:
            explained_variance = float(
                (1.0 - torch.var(batch.returns - batch.values, unbiased=False) / returns_var).item()
            )

        batch_size = batch.actions.size(0)
        minibatch_size = max(batch_size // config.ppo.minibatches, 1)
        train_metrics = MetricAggregator()

        for _epoch in range(config.ppo.update_epochs):
            indices = torch.randperm(batch_size, device=ctx.device)
            for start in range(0, batch_size, minibatch_size):
                batch_index = indices[start : start + minibatch_size]
                obs_batch = _state_index(batch.obs, batch_index)
                state_batch = _state_index(batch.states, batch_index)
                done_batch = batch.done_inputs[batch_index]
                action_batch = batch.actions[batch_index]
                old_log_prob = batch.log_probs[batch_index]
                old_value = batch.values[batch_index]
                return_batch = batch.returns[batch_index]
                adv_batch = batch.advantages[batch_index]

                with _autocast_context(ctx):
                    outputs = policy_evaluate_actions(
                        model,
                        obs_batch,
                        action_batch,
                        state=state_batch,
                        done=done_batch,
                    )
                    log_ratio = outputs["log_prob"] - old_log_prob
                    ratio = log_ratio.exp()
                    unclipped = adv_batch * ratio
                    clipped = adv_batch * torch.clamp(ratio, 1.0 - config.ppo.clip_coef, 1.0 + config.ppo.clip_coef)
                    policy_loss = -torch.min(unclipped, clipped).mean()

                    new_value = outputs["value"]
                    value_clipped = old_value + torch.clamp(
                        new_value - old_value,
                        -config.ppo.value_clip_coef,
                        config.ppo.value_clip_coef,
                    )
                    value_loss_unclipped = (new_value - return_batch).pow(2)
                    value_loss_clipped = (value_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    entropy = outputs["entropy"].mean()
                    aux_loss = sum(outputs["aux_losses"].values()) if outputs["aux_losses"] else torch.zeros((), device=ctx.device)
                    loss = policy_loss + config.ppo.vf_coef * value_loss - config.ppo.ent_coef * entropy + aux_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.ppo.max_grad_norm)
                optimizer.step()

                approx_kl = ((ratio - 1.0) - log_ratio).mean().abs()
                train_metrics.update(
                    {
                        "loss": loss.detach(),
                        "policy_loss": policy_loss.detach(),
                        "value_loss": value_loss.detach(),
                        "entropy": entropy.detach(),
                        "approx_kl": approx_kl.detach(),
                        **scalarize_metrics(outputs["metrics"]),
                    }
                )
                if config.ppo.target_kl is not None and approx_kl.item() > config.ppo.target_kl:
                    break

        elapsed = max(time.perf_counter() - start_time, 1e-6)
        aggregate_metrics = reduce_scalar_dict(
            {
                "global_step": float(global_step),
                "wall_clock_seconds": elapsed,
                "throughput_fps": global_step / elapsed,
                "gpu_utilization": try_get_gpu_utilization() or 0.0,
                "explained_variance": explained_variance,
                **rollout_metrics.compute(),
                **train_metrics.compute(),
            },
            ctx,
        )
        episode_sums = reduce_scalar_dict(
            {
                "return_sum": float(np.sum(update_completed_returns)) if update_completed_returns else 0.0,
                "success_sum": float(np.sum(update_completed_successes)) if update_completed_successes else 0.0,
                "length_sum": float(np.sum(update_completed_lengths)) if update_completed_lengths else 0.0,
                "episode_count": float(len(update_completed_returns)),
            },
            ctx,
            average=False,
        )
        episode_count = max(episode_sums["episode_count"], 1.0)
        update_metrics = {
            **aggregate_metrics,
            "train/episode_return": episode_sums["return_sum"] / episode_count if episode_sums["episode_count"] > 0 else 0.0,
            "train/success_rate": episode_sums["success_sum"] / episode_count if episode_sums["episode_count"] > 0 else 0.0,
            "train/episode_length": episode_sums["length_sum"] / episode_count if episode_sums["episode_count"] > 0 else 0.0,
            "train/episodes_completed": episode_sums["episode_count"],
        }

        if ctx.is_main_process and (update % config.system.log_interval == 0 or update == total_updates):
            LOGGER.info(
                "update=%s step=%s return=%.3f success=%.3f throughput=%.1f",
                update,
                global_step,
                update_metrics["train/episode_return"],
                update_metrics["train/success_rate"],
                update_metrics["throughput_fps"],
            )
            logger.log(update, update_metrics)

        if update % config.system.checkpoint_interval == 0 or update == total_updates:
            latest = save_checkpoint(config, model, optimizer, update, global_step, ctx.is_main_process)
            if latest is not None:
                latest_checkpoint = latest

        barrier(ctx)

    eval_metrics = evaluate_policy(config, unwrap_ddp(model), ctx)
    final_metrics = {**update_metrics, **eval_metrics}
    if ctx.is_main_process:
        logger.log(total_updates, final_metrics)
        summary = "\n".join(
            [
                "# Training Summary",
                "",
                f"- env: `{config.env.env_id}`",
                f"- variant: `{config.model.variant}`",
                f"- global_step: `{int(final_metrics['global_step'])}`",
                f"- train_return: `{final_metrics['train/episode_return']:.3f}`",
                f"- eval_return: `{final_metrics['eval_return']:.3f}`",
                f"- eval_success_rate: `{final_metrics['eval_success_rate']:.3f}`",
            ]
        )
        logger.write_summary(summary)
        logger.close()
    envs.close()
    return TrainResult(
        output_dir=config.logging.output_dir,
        latest_checkpoint=latest_checkpoint,
        final_metrics=final_metrics,
    )
