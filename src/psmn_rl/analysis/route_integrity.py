from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from psmn_rl.config import load_config
from psmn_rl.envs.registry import make_eval_env, make_reset_seeds, make_vector_env
from psmn_rl.models.factory import build_model
from psmn_rl.rl.distributed.ddp import cleanup_distributed, init_distributed
from psmn_rl.rl.ppo.algorithm import _episode_successes, prepare_done, prepare_obs
from psmn_rl.utils.seed import capture_rng_state, restore_rng_state, set_seed


def _discover_run_dirs(paths: list[str]) -> list[Path]:
    discovered: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if (path / "resolved_config.yaml").exists() and (path / "latest.pt").exists():
            discovered.append(path)
            continue
        discovered.extend(sorted(parent for parent in path.glob("*/resolved_config.yaml") if (parent.parent / "latest.pt").exists()))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in discovered:
        run_dir = path if path.is_dir() else path.parent
        resolved = run_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(run_dir)
    return unique


def _scalar_metric(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().cpu().item())
    return float(value)


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _evaluate_run(run_dir: Path, ctx, episodes: int) -> dict[str, Any]:
    config = load_config(run_dir / "resolved_config.yaml")
    config.system.device = str(ctx.device)
    set_seed(config.seed + ctx.rank, deterministic=config.system.deterministic)
    envs = make_vector_env(config.env, seed=config.seed, world_rank=ctx.rank)
    model = build_model(config.model, envs.single_observation_space, envs.single_action_space).to(ctx.device)
    envs.close()
    checkpoint = torch.load(run_dir / "latest.pt", map_location=ctx.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    was_training = model.training
    model.eval()
    try:
        if not ctx.is_main_process:
            return {}
        rng_state = capture_rng_state()
        set_seed(config.seed + 444_444, deterministic=config.system.deterministic)
        eval_env = make_eval_env(config.env, seed=config.seed + 444, world_rank=0)
        eval_seed = make_reset_seeds(config.env.num_eval_envs, config.seed + 444, world_rank=0)
        obs, _ = eval_env.reset(seed=eval_seed)
        obs_t = prepare_obs(obs, ctx.device)
        done_t = torch.ones(config.env.num_eval_envs, device=ctx.device, dtype=torch.bool)
        state = model.initial_state(config.env.num_eval_envs, ctx.device)
        returns = np.zeros(config.env.num_eval_envs, dtype=np.float32)
        lengths = np.zeros(config.env.num_eval_envs, dtype=np.int32)
        finished_returns: list[float] = []
        finished_successes: list[float] = []
        step_count = 0
        metric_sums: dict[str, float] = {}
        try:
            while len(finished_returns) < episodes:
                with torch.inference_mode():
                    output = model(obs_t, state=state, done=done_t)
                    action = output.logits.argmax(dim=-1)
                    state = output.next_state
                for key, value in output.metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + _scalar_metric(value)
                step_count += 1
                next_obs, reward, terminated, truncated, info = eval_env.step(action.detach().cpu().numpy())
                done_np = np.logical_or(terminated, truncated)
                success = _episode_successes(reward, done_np, info)
                returns += reward
                lengths += 1
                for index, finished in enumerate(done_np):
                    if not finished:
                        continue
                    finished_returns.append(float(returns[index]))
                    finished_successes.append(float(success[index]))
                    returns[index] = 0.0
                    lengths[index] = 0
                obs_t = prepare_obs(next_obs, ctx.device)
                done_t = prepare_done(done_np, ctx.device)
        finally:
            eval_env.close()
            restore_rng_state(rng_state)
        metrics = {
            "run_name": run_dir.name,
            "variant": config.model.variant,
            "eval_return": float(np.mean(finished_returns[:episodes])),
            "eval_success_rate": float(np.mean(finished_successes[:episodes])),
        }
        for key, value in sorted(metric_sums.items()):
            metrics[key] = value / max(step_count, 1)
        return metrics
    finally:
        if was_training:
            model.train()


def _build_report(rows: list[dict[str, Any]], episodes: int) -> str:
    expert_keys = sorted({key for row in rows for key in row if key.startswith("expert_load_")})
    header = [
        "Run",
        "Variant",
        "Eval Success",
        "Eval Return",
        "Route Entropy",
        "Path Entropy",
        "Active Compute",
        *expert_keys,
    ]
    lines = [
        "# Distilled Route Integrity Report",
        "",
        f"- episodes: `{episodes}`",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---", "---", "---:", "---:", "---:", "---:", "---:", *(["---:"] * len(expert_keys))]) + " |",
    ]
    for row in rows:
        values = [
            str(row["run_name"]),
            str(row["variant"]),
            _format_float(row.get("eval_success_rate")),
            _format_float(row.get("eval_return")),
            _format_float(row.get("route_entropy")),
            _format_float(row.get("path_entropy")),
            _format_float(row.get("active_compute_proxy")),
            *[_format_float(row.get(key)) for key in expert_keys],
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Interpretation", ""])
    if len(rows) >= 2:
        baseline = rows[0]
        candidate = rows[1]
        lines.append(
            f"- `{candidate['run_name']}` changes greedy success from `{baseline.get('eval_success_rate', 0.0):.4f}` "
            f"to `{candidate.get('eval_success_rate', 0.0):.4f}`."
        )
        lines.append(
            f"- Route entropy moves from `{baseline.get('route_entropy', 0.0):.4f}` to "
            f"`{candidate.get('route_entropy', 0.0):.4f}`, and active compute proxy moves from "
            f"`{baseline.get('active_compute_proxy', 0.0):.4f}` to `{candidate.get('active_compute_proxy', 0.0):.4f}`."
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare route metrics across greedy-evaluated run directories.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--csv", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dirs = _discover_run_dirs(args.paths)
    if not run_dirs:
        raise SystemExit("no run directories found")
    ctx = init_distributed(args.device, "auto")
    try:
        rows = [_evaluate_run(run_dir, ctx, args.episodes) for run_dir in run_dirs]
        if not ctx.is_main_process:
            return
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_build_report(rows, args.episodes), encoding="utf-8")
        if args.csv is not None:
            csv_path = Path(args.csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = sorted({key for row in rows for key in row.keys()})
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
