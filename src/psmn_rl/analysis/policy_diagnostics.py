from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from psmn_rl.config import load_config
from psmn_rl.envs.registry import make_vector_env
from psmn_rl.logging import configure_logging
from psmn_rl.models.factory import build_model
from psmn_rl.rl.distributed.ddp import cleanup_distributed, init_distributed
from psmn_rl.rl.ppo.algorithm import collect_policy_diagnostics
from psmn_rl.utils.seed import set_seed


DEFAULT_MODES = [
    ("greedy", True, 1.0),
    ("sampled_t1.0", False, 1.0),
    ("sampled_t0.7", False, 0.7),
    ("sampled_t0.5", False, 0.5),
]


def _load_final_train_metrics(run_dir: Path) -> dict[str, float]:
    metrics_path = run_dir / "metrics.jsonl"
    final: dict[str, float] = {}
    if not metrics_path.exists():
        return final
    for line in metrics_path.read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            final = row
    return {key: value for key, value in final.items() if isinstance(value, (int, float))}


def _discover_run_dirs(paths: list[str]) -> list[Path]:
    discovered: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if (path / "resolved_config.yaml").exists():
            discovered.append(path)
            continue
        discovered.extend(sorted(parent for parent in path.glob("*/resolved_config.yaml") for parent in [parent.parent]))
    unique = []
    seen: set[Path] = set()
    for path in discovered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _build_report(rows: list[dict[str, Any]], run_dirs: list[Path], episodes: int) -> str:
    lines = [
        "# Policy Extraction Report",
        "",
        f"- episodes per mode: `{episodes}`",
        f"- run count: `{len(run_dirs)}`",
        "",
    ]
    lines.append("## Greedy vs Best Sampled")
    lines.append("")
    lines.append("| Variant | Greedy Success | Best Sampled Success | Best Sampled Mode | Greedy Max Prob | Greedy Margin | Best Sampled Greedy-Match |")
    lines.append("| --- | ---: | ---: | --- | ---: | ---: | ---: |")
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    for variant, variant_rows in sorted(by_variant.items()):
        greedy_row = next(row for row in variant_rows if row["mode"] == "greedy")
        sampled_rows = [row for row in variant_rows if row["mode"] != "greedy"]
        best_sampled = max(sampled_rows, key=lambda row: row.get("eval_success_rate", 0.0)) if sampled_rows else greedy_row
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    _format_float(greedy_row.get("eval_success_rate")),
                    _format_float(best_sampled.get("eval_success_rate")),
                    str(best_sampled["mode"]),
                    _format_float(greedy_row.get("eval/action_max_prob")),
                    _format_float(greedy_row.get("eval/action_logit_margin")),
                    _format_float(best_sampled.get("eval/action_greedy_match")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Mode Table", "", "| Variant | Mode | Eval Return | Eval Success | Eval Entropy | Eval Max Prob | Eval Margin | Eval Greedy Match | Train Return | Train Success | Train Entropy | Train Max Prob | Throughput |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    sort_rows = sorted(rows, key=lambda row: (str(row["variant"]), str(row["mode"])))
    for row in sort_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    str(row["mode"]),
                    _format_float(row.get("eval_return")),
                    _format_float(row.get("eval_success_rate")),
                    _format_float(row.get("eval/action_entropy")),
                    _format_float(row.get("eval/action_max_prob")),
                    _format_float(row.get("eval/action_logit_margin")),
                    _format_float(row.get("eval/action_greedy_match")),
                    _format_float(row.get("train/episode_return")),
                    _format_float(row.get("train/success_rate")),
                    _format_float(row.get("rollout/action_entropy")),
                    _format_float(row.get("rollout/action_max_prob")),
                    _format_float(row.get("throughput_fps")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run compact policy-extraction diagnostics over training run directories.")
    parser.add_argument("paths", nargs="+", help="Run directories or parent directories containing run directories")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--trace-dir", type=str, default=None)
    parser.add_argument("--trace-limit", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dirs = _discover_run_dirs(args.paths)
    if not run_dirs:
        raise SystemExit("no run directories found")
    ctx = init_distributed(args.device, "auto")
    configure_logging(ctx.is_main_process)
    rows: list[dict[str, Any]] = []
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    if trace_dir is not None and ctx.is_main_process:
        trace_dir.mkdir(parents=True, exist_ok=True)
    try:
        for run_dir in run_dirs:
            config = load_config(run_dir / "resolved_config.yaml")
            config.system.device = args.device
            config.logging.tensorboard = False
            set_seed(config.seed + ctx.rank, deterministic=config.system.deterministic)
            envs = make_vector_env(config.env, seed=config.seed, world_rank=ctx.rank)
            model = build_model(config.model, envs.single_observation_space, envs.single_action_space).to(ctx.device)
            envs.close()
            checkpoint = torch.load(run_dir / "latest.pt", map_location=ctx.device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            train_metrics = _load_final_train_metrics(run_dir)
            for mode_name, greedy, temperature in DEFAULT_MODES:
                diagnostics = collect_policy_diagnostics(
                    config=config,
                    model=model,
                    ctx=ctx,
                    episodes=args.episodes,
                    greedy=greedy,
                    temperature=temperature,
                    trace_limit=args.trace_limit,
                )
                row = {
                    "run_dir": str(run_dir),
                    "variant": config.model.variant,
                    "env_id": config.env.env_id,
                    "mode": mode_name,
                    "greedy": greedy,
                    "temperature": temperature,
                    **train_metrics,
                    **diagnostics.metrics,
                }
                rows.append(row)
                if trace_dir is not None and ctx.is_main_process:
                    trace_path = trace_dir / f"{run_dir.name}_{mode_name}.json"
                    trace_path.write_text(json.dumps(diagnostics.episodes, indent=2, sort_keys=True))
        if not ctx.is_main_process:
            return
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_build_report(rows, run_dirs, args.episodes), encoding="utf-8")
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
