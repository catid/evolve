from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from psmn_rl.analysis.policy_diagnostics import DEFAULT_MODES
from psmn_rl.config import dump_config, load_config
from psmn_rl.envs.registry import make_eval_env, make_reset_seeds, make_vector_env
from psmn_rl.models.common import ActorCriticModel
from psmn_rl.models.factory import build_model
from psmn_rl.rl.distributed.ddp import DistributedContext, detect_device
from psmn_rl.rl.ppo.algorithm import _episode_successes, collect_policy_diagnostics, policy_act, prepare_done, prepare_obs
from psmn_rl.utils.seed import set_seed


@dataclass(slots=True)
class TrajectoryBatch:
    obs: dict[str, torch.Tensor]
    actions: torch.Tensor
    weights: torch.Tensor
    successes: int
    episodes_seen: int
    steps: int
    mean_return: float
    mean_length: float


def _local_ctx(device: torch.device) -> DistributedContext:
    return DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=device,
        is_distributed=False,
        is_main_process=True,
        autocast_dtype=None,
    )


def _clone_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in obs.items()}


def _stack_obs(samples: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    first = samples[0]
    return {
        key: torch.as_tensor(np.stack([sample[key] for sample in samples], axis=0))
        for key in first
    }


def _set_trainable_parameters(model: ActorCriticModel, target: str) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.policy_head.parameters():
        parameter.requires_grad_(True)
    if target == "policy_head_plus_last_shared":
        core = model.core
        if hasattr(core, "blocks") and getattr(core, "blocks"):
            for parameter in core.blocks[-1].parameters():
                parameter.requires_grad_(True)
            if hasattr(core, "norm"):
                for parameter in core.norm.parameters():
                    parameter.requires_grad_(True)
        elif hasattr(core, "bank"):
            for attr in ("query", "bank", "output_norm"):
                module = getattr(core, attr, None)
                if module is None:
                    continue
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            expert_keys = getattr(core, "expert_keys", None)
            if isinstance(expert_keys, torch.nn.Parameter):
                expert_keys.requires_grad_(True)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def harvest_successful_trajectories(
    config_path: str,
    checkpoint_path: str,
    device: str,
    temperature: float,
    success_target: int,
    episode_cap: int,
) -> TrajectoryBatch:
    config = load_config(config_path)
    config.system.device = device
    set_seed(config.seed + 10_001, deterministic=config.system.deterministic)
    eval_env = make_eval_env(config.env, seed=config.seed + 999, world_rank=0)
    reset_seed = make_reset_seeds(config.env.num_eval_envs, config.seed + 999, world_rank=0)
    obs, _ = eval_env.reset(seed=reset_seed)
    envs = make_vector_env(config.env, seed=config.seed)
    model = build_model(config.model, envs.single_observation_space, envs.single_action_space)
    envs.close()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"])
    device_t = detect_device(device)
    model.to(device_t)
    model.eval()
    obs_t = prepare_obs(obs, device_t)
    done_t = torch.ones(config.env.num_eval_envs, device=device_t, dtype=torch.bool)
    state = model.initial_state(config.env.num_eval_envs, device_t)

    episode_obs: list[list[dict[str, np.ndarray]]] = [[] for _ in range(config.env.num_eval_envs)]
    episode_actions: list[list[int]] = [[] for _ in range(config.env.num_eval_envs)]
    episode_returns = np.zeros(config.env.num_eval_envs, dtype=np.float32)
    episode_lengths = np.zeros(config.env.num_eval_envs, dtype=np.int32)
    successful_obs: list[dict[str, np.ndarray]] = []
    successful_actions: list[int] = []
    successful_weights: list[float] = []
    successful_returns: list[float] = []
    successful_lengths: list[float] = []
    episodes_seen = 0

    try:
        while len(successful_returns) < success_target and episodes_seen < episode_cap:
            with torch.inference_mode():
                action, _, _, state, _, _ = policy_act(
                    model,
                    obs_t,
                    state=state,
                    done=done_t,
                    greedy=False,
                    temperature=temperature,
                )
            action_np = action.detach().cpu().numpy()
            raw_obs = {
                key: np.asarray(value)
                for key, value in obs.items()
                if key in ("image", "direction", "pixels")
            }
            for index in range(config.env.num_eval_envs):
                episode_obs[index].append({key: value[index].copy() for key, value in raw_obs.items()})
                episode_actions[index].append(int(action_np[index]))
            next_obs, reward, terminated, truncated, info = eval_env.step(action_np)
            done_np = np.logical_or(terminated, truncated)
            step_success = _episode_successes(reward, done_np, info)
            episode_returns += reward
            episode_lengths += 1
            for index, finished in enumerate(done_np):
                if not finished:
                    continue
                episodes_seen += 1
                if step_success[index] > 0.0:
                    episode_return = float(episode_returns[index])
                    successful_returns.append(episode_return)
                    successful_lengths.append(float(episode_lengths[index]))
                    successful_obs.extend(episode_obs[index])
                    successful_actions.extend(episode_actions[index])
                    successful_weights.extend([episode_return] * len(episode_actions[index]))
                episode_obs[index].clear()
                episode_actions[index].clear()
                episode_returns[index] = 0.0
                episode_lengths[index] = 0
            obs = next_obs
            obs_t = prepare_obs(obs, device_t)
            done_t = prepare_done(done_np, device_t)
    finally:
        eval_env.close()

    if not successful_returns:
        raise RuntimeError("no successful sampled trajectories were harvested")
    return TrajectoryBatch(
        obs=_stack_obs(successful_obs),
        actions=torch.as_tensor(successful_actions, dtype=torch.long),
        weights=torch.as_tensor(successful_weights, dtype=torch.float32),
        successes=len(successful_returns),
        episodes_seen=episodes_seen,
        steps=len(successful_actions),
        mean_return=float(np.mean(successful_returns)),
        mean_length=float(np.mean(successful_lengths)),
    )


def _weighted_cross_entropy(logits: torch.Tensor, actions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits, actions, reduction="none")
    norm = weights.mean().clamp_min(1e-8)
    return (losses * (weights / norm)).mean()


def fine_tune_from_trajectories(
    config_path: str,
    checkpoint_path: str,
    device: str,
    target: str,
    weighting: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    trajectories: TrajectoryBatch,
) -> tuple[ActorCriticModel, dict[str, float]]:
    config = load_config(config_path)
    config.system.device = device
    set_seed(config.seed + 10_002, deterministic=config.system.deterministic)
    envs = make_vector_env(config.env, seed=config.seed)
    model = build_model(config.model, envs.single_observation_space, envs.single_action_space)
    envs.close()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"])
    device_t = detect_device(device)
    model.to(device_t)
    trainable_params = _set_trainable_parameters(model, target)
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=learning_rate)

    obs = {key: value.to(device_t) for key, value in trajectories.obs.items()}
    actions = trajectories.actions.to(device_t)
    if weighting == "return":
        weights = trajectories.weights.to(device_t)
    else:
        weights = torch.ones_like(trajectories.weights, device=device_t)
    done = torch.zeros(actions.size(0), device=device_t, dtype=torch.bool)
    epoch_losses: list[float] = []

    model.train()
    for _epoch in range(epochs):
        indices = torch.randperm(actions.size(0), device=device_t)
        for start in range(0, actions.size(0), batch_size):
            batch_index = indices[start : start + batch_size]
            obs_batch = {key: value[batch_index] for key, value in obs.items()}
            done_batch = done[batch_index]
            action_batch = actions[batch_index]
            weight_batch = weights[batch_index]
            output = model(obs_batch, state={}, done=done_batch)
            loss = _weighted_cross_entropy(output.logits, action_batch, weight_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

    return model, {
        "fine_tune/steps": float(len(epoch_losses)),
        "fine_tune/loss_final": float(epoch_losses[-1]) if epoch_losses else 0.0,
        "fine_tune/loss_mean": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
        "fine_tune/trainable_params": float(trainable_params),
    }


def evaluate_modes(config_path: str, model: ActorCriticModel, device: str, episodes: int) -> dict[str, dict[str, float]]:
    config = load_config(config_path)
    config.system.device = device
    ctx = _local_ctx(detect_device(device))
    set_seed(config.seed, deterministic=config.system.deterministic)
    metrics: dict[str, dict[str, float]] = {}
    for mode_name, greedy, temperature in DEFAULT_MODES:
        diagnostics = collect_policy_diagnostics(
            config=config,
            model=model,
            ctx=ctx,
            episodes=episodes,
            greedy=greedy,
            temperature=temperature,
            trace_limit=0,
        )
        metrics[mode_name] = diagnostics.metrics
    return metrics


def _best_sampled_mode(metrics_by_mode: dict[str, dict[str, float]]) -> tuple[str, float]:
    sampled = [(mode, metrics) for mode, metrics in metrics_by_mode.items() if mode != "greedy"]
    best_mode, best_metrics = max(sampled, key=lambda item: item[1].get("eval_success_rate", 0.0))
    return best_mode, float(best_metrics.get("eval_success_rate", 0.0))


def _write_single_run_summary(
    output_dir: Path,
    variant: str,
    target: str,
    weighting: str,
    teacher_temperature: float,
    harvest: TrajectoryBatch,
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
    fine_tune_metrics: dict[str, float],
) -> None:
    before_best_mode, before_best_sampled = _best_sampled_mode(before)
    after_best_mode, after_best_sampled = _best_sampled_mode(after)
    summary = {
        "variant": variant,
        "target": target,
        "weighting": weighting,
        "teacher_temperature": teacher_temperature,
        "harvest_successes": harvest.successes,
        "harvest_episodes_seen": harvest.episodes_seen,
        "harvest_steps": harvest.steps,
        "harvest_mean_return": harvest.mean_return,
        "harvest_mean_length": harvest.mean_length,
        "before_greedy_success": float(before["greedy"].get("eval_success_rate", 0.0)),
        "before_greedy_return": float(before["greedy"].get("eval_return", 0.0)),
        "before_greedy_max_prob": float(before["greedy"].get("eval/action_max_prob", 0.0)),
        "before_greedy_margin": float(before["greedy"].get("eval/action_logit_margin", 0.0)),
        "before_best_sampled_mode": before_best_mode,
        "before_best_sampled_success": before_best_sampled,
        "after_greedy_success": float(after["greedy"].get("eval_success_rate", 0.0)),
        "after_greedy_return": float(after["greedy"].get("eval_return", 0.0)),
        "after_greedy_max_prob": float(after["greedy"].get("eval/action_max_prob", 0.0)),
        "after_greedy_margin": float(after["greedy"].get("eval/action_logit_margin", 0.0)),
        "after_best_sampled_mode": after_best_mode,
        "after_best_sampled_success": after_best_sampled,
        **fine_tune_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Self-Imitation Summary",
                "",
                f"- variant: `{variant}`",
                f"- target: `{target}`",
                f"- weighting: `{weighting}`",
                f"- teacher_temperature: `{teacher_temperature}`",
                f"- harvest_successes: `{harvest.successes}`",
                f"- before_greedy_success: `{summary['before_greedy_success']:.3f}`",
                f"- after_greedy_success: `{summary['after_greedy_success']:.3f}`",
                f"- before_best_sampled_mode: `{before_best_mode}`",
                f"- after_best_sampled_mode: `{after_best_mode}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _discover_run_dirs(paths: list[str]) -> list[Path]:
    discovered: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if (path / "summary.json").exists():
            discovered.append(path)
            continue
        discovered.extend(sorted(parent.parent for parent in path.glob("*/summary.json")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in discovered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _build_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Self-Imitation Report",
        "",
        "## Summary",
        "",
        "| Run | Variant | Target | Weighting | Harvest Successes | Before Greedy Success | After Greedy Success | Before Best Sampled | After Best Sampled | Trainable Params |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (str(item["variant"]), str(item["target"]), str(item["weighting"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['run_name']}`",
                    str(row["variant"]),
                    str(row["target"]),
                    str(row["weighting"]),
                    str(int(row["harvest_successes"])),
                    f"{row['before_greedy_success']:.3f}",
                    f"{row['after_greedy_success']:.3f}",
                    f"{row['before_best_sampled_success']:.3f}",
                    f"{row['after_best_sampled_success']:.3f}",
                    f"{row['fine_tune/trainable_params']:.0f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["variant"]), str(item["target"]), str(item["weighting"]))):
        lines.append(
            f"- `{row['run_name']}` moves greedy success from `{row['before_greedy_success']:.3f}` "
            f"to `{row['after_greedy_success']:.3f}` while sampled success moves from "
            f"`{row['before_best_sampled_success']:.3f}` to `{row['after_best_sampled_success']:.3f}`."
        )
    return "\n".join(lines) + "\n"


def run_once(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    dump_config(config, output_dir / "resolved_config.yaml")
    before_model_env = make_vector_env(config.env, seed=config.seed)
    before_model = build_model(config.model, before_model_env.single_observation_space, before_model_env.single_action_space)
    before_model_env.close()
    before_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"])
    before_model.to(detect_device(args.device))
    before = evaluate_modes(args.config, before_model, args.device, args.eval_episodes)
    harvest = harvest_successful_trajectories(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        temperature=args.teacher_temperature,
        success_target=args.success_target,
        episode_cap=args.max_episodes,
    )
    torch.save(
        {
            "obs": harvest.obs,
            "actions": harvest.actions,
            "weights": harvest.weights,
            "successes": harvest.successes,
            "episodes_seen": harvest.episodes_seen,
            "steps": harvest.steps,
            "mean_return": harvest.mean_return,
            "mean_length": harvest.mean_length,
        },
        output_dir / "harvest.pt",
    )
    model, fine_tune_metrics = fine_tune_from_trajectories(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        target=args.target,
        weighting=args.weighting,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        trajectories=harvest,
    )
    after = evaluate_modes(args.config, model, args.device, args.eval_episodes)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint["model"] = model.state_dict()
    checkpoint["self_imitation"] = {
        "target": args.target,
        "weighting": args.weighting,
        "teacher_temperature": args.teacher_temperature,
        "success_target": args.success_target,
        **fine_tune_metrics,
    }
    torch.save(checkpoint, output_dir / "latest.pt")
    _write_single_run_summary(
        output_dir=output_dir,
        variant=config.model.variant,
        target=args.target,
        weighting=args.weighting,
        teacher_temperature=args.teacher_temperature,
        harvest=harvest,
        before=before,
        after=after,
        fine_tune_metrics=fine_tune_metrics,
    )


def build_report(args: argparse.Namespace) -> None:
    run_dirs = _discover_run_dirs(args.paths)
    if not run_dirs:
        raise SystemExit("no self-imitation runs found")
    rows = []
    for run_dir in run_dirs:
        row = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        row["run_name"] = run_dir.name
        rows.append(row)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_build_report(rows), encoding="utf-8")
    if args.csv is not None:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run self-imitation fine-tuning from successful sampled trajectories.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--checkpoint", required=True)
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--device", default="cpu")
    run_parser.add_argument("--teacher-temperature", type=float, default=1.0)
    run_parser.add_argument("--success-target", type=int, default=64)
    run_parser.add_argument("--max-episodes", type=int, default=256)
    run_parser.add_argument("--target", choices=("policy_head", "policy_head_plus_last_shared"), default="policy_head")
    run_parser.add_argument("--weighting", choices=("uniform", "return"), default="uniform")
    run_parser.add_argument("--batch-size", type=int, default=128)
    run_parser.add_argument("--epochs", type=int, default=8)
    run_parser.add_argument("--learning-rate", type=float, default=1e-4)
    run_parser.add_argument("--eval-episodes", type=int, default=32)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("paths", nargs="+")
    report_parser.add_argument("--output", required=True)
    report_parser.add_argument("--csv", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run_once(args)
        return
    build_report(args)


if __name__ == "__main__":
    main()
