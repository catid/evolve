from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from psmn_rl.config import ExperimentConfig
from psmn_rl.envs.registry import make_eval_env, make_vector_env
from psmn_rl.logging import LOGGER, RunLogger
from psmn_rl.metrics import MetricAggregator, scalarize_metrics
from psmn_rl.models.common import ActorCriticModel
from psmn_rl.rl.distributed.ddp import DistributedContext, barrier, reduce_scalar_dict, unwrap_ddp
from psmn_rl.rl.rollout.storage import RolloutStorage
from psmn_rl.utils.io import get_git_commit, save_json, try_get_gpu_utilization


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
        "step": step,
        "config": config.to_dict(),
        "git_commit": get_git_commit(),
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
    eval_env = make_eval_env(config.env, seed=config.seed + 999)
    obs, _ = eval_env.reset(seed=config.seed + 999)
    obs_t = prepare_obs(obs, ctx.device)
    done_t = torch.ones(config.env.num_eval_envs, device=ctx.device, dtype=torch.bool)
    state = model.initial_state(config.env.num_eval_envs, ctx.device)
    returns = np.zeros(config.env.num_eval_envs, dtype=np.float32)
    lengths = np.zeros(config.env.num_eval_envs, dtype=np.int32)
    finished_returns: list[float] = []
    finished_successes: list[float] = []
    target_episodes = episodes or config.evaluation.episodes
    while len(finished_returns) < target_episodes:
        with torch.no_grad():
            action, _, _, state, _, _ = policy_act(
                model,
                obs_t,
                state=state,
                done=done_t,
                greedy=config.evaluation.greedy,
            )
        next_obs, reward, terminated, truncated, _ = eval_env.step(action.cpu().numpy())
        done_np = np.logical_or(terminated, truncated)
        returns += reward
        lengths += 1
        for index, finished in enumerate(done_np):
            if finished:
                finished_returns.append(float(returns[index]))
                finished_successes.append(float(returns[index] > 0.0))
                returns[index] = 0.0
                lengths[index] = 0
        obs_t = prepare_obs(next_obs, ctx.device)
        done_t = prepare_done(done_np, ctx.device)
    eval_env.close()
    metrics = {
        "eval_return": float(np.mean(finished_returns[:target_episodes])),
        "eval_success_rate": float(np.mean(finished_successes[:target_episodes])),
    }
    return metrics


def train(
    config: ExperimentConfig,
    model: ActorCriticModel,
    optimizer: torch.optim.Optimizer,
    ctx: DistributedContext,
    max_updates: int | None = None,
) -> TrainResult:
    logger = RunLogger(config, enabled=ctx.is_main_process)
    envs = make_vector_env(config.env, seed=config.seed, world_rank=ctx.rank)
    obs, _ = envs.reset(seed=config.seed + ctx.rank)
    obs_t = prepare_obs(obs, ctx.device)
    done_t = torch.ones(config.env.num_envs, device=ctx.device, dtype=torch.bool)
    state = unwrap_ddp(model).initial_state(config.env.num_envs, ctx.device)

    total_updates = max_updates or config.ppo.total_updates
    global_step = 0
    start_time = time.perf_counter()
    episode_returns = np.zeros(config.env.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(config.env.num_envs, dtype=np.int32)
    completed_returns: list[float] = []
    completed_successes: list[float] = []
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

    for update in range(1, total_updates + 1):
        if config.ppo.anneal_lr:
            frac = 1.0 - (update - 1) / max(total_updates, 1)
            optimizer.param_groups[0]["lr"] = frac * config.ppo.learning_rate

        rollout = RolloutStorage()
        rollout_metrics = MetricAggregator()

        for _ in range(config.ppo.rollout_steps):
            with torch.no_grad():
                with _autocast_context(ctx):
                    action, log_prob, value, next_state, metrics, _ = policy_act(
                        model,
                        obs_t,
                        state=state,
                        done=done_t,
                    )

            next_obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
            reward_t = torch.as_tensor(reward, device=ctx.device, dtype=torch.float32)
            next_done_np = np.logical_or(terminated, truncated)
            next_done_t = prepare_done(next_done_np, ctx.device)
            rollout.add(obs_t, state, done_t, action, log_prob, value, reward_t, next_done_t)
            rollout_metrics.update(metrics)

            episode_returns += reward
            episode_lengths += 1
            for env_index, finished in enumerate(next_done_np):
                if finished:
                    completed_returns.append(float(episode_returns[env_index]))
                    completed_successes.append(float(episode_returns[env_index] > 0.0))
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
        update_metrics = {
            "global_step": float(global_step),
            "wall_clock_seconds": elapsed,
            "throughput_fps": global_step / elapsed,
            "train/episode_return": float(np.mean(completed_returns[-32:])) if completed_returns else 0.0,
            "train/success_rate": float(np.mean(completed_successes[-32:])) if completed_successes else 0.0,
            "gpu_utilization": try_get_gpu_utilization() or 0.0,
            **rollout_metrics.compute(),
            **train_metrics.compute(),
        }
        update_metrics = reduce_scalar_dict(update_metrics, ctx)

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
            latest = save_checkpoint(config, model, optimizer, global_step, ctx.is_main_process)
            if latest is not None:
                latest_checkpoint = latest

        barrier(ctx)

    eval_metrics = evaluate_policy(config, unwrap_ddp(model), ctx)
    eval_metrics = reduce_scalar_dict(eval_metrics, ctx)
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
