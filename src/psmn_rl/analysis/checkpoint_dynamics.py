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


def _discover_run_dirs(paths: list[str]) -> list[Path]:
    discovered: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if (path / "resolved_config.yaml").exists():
            discovered.append(path)
            continue
        discovered.extend(sorted(parent.parent for parent in path.glob("*/resolved_config.yaml")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for run_dir in discovered:
        resolved = run_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(run_dir)
    return unique


def _load_train_metrics_by_update(run_dir: Path) -> dict[int, dict[str, float]]:
    metrics_path = run_dir / "metrics.jsonl"
    train_by_update: dict[int, dict[str, float]] = {}
    if not metrics_path.exists():
        return train_by_update
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("type") != "scalar":
            continue
        update = int(row["step"])
        train_by_update[update] = {
            key: float(value)
            for key, value in row.items()
            if key not in {"type", "step"} and isinstance(value, (int, float))
        }
    return train_by_update


def _checkpoint_rows(run_dir: Path) -> list[tuple[int, Path]]:
    rows: list[tuple[int, Path]] = []
    for checkpoint_path in sorted(run_dir.glob("checkpoint_update_*.pt")):
        stem = checkpoint_path.stem
        update = int(stem.rsplit("_", 1)[-1])
        rows.append((update, checkpoint_path))
    latest_path = run_dir / "latest.pt"
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        latest_update = int(checkpoint.get("update", 0))
        if latest_update not in {update for update, _ in rows}:
            rows.append((latest_update, latest_path))
    return sorted(rows, key=lambda item: item[0])


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _build_report(rows: list[dict[str, Any]], run_dirs: list[Path], episodes: int) -> str:
    lines = [
        "# Checkpoint Dynamics Report",
        "",
        f"- episodes per mode: `{episodes}`",
        f"- run count: `{len(run_dirs)}`",
        "",
        "## Best Checkpoints",
        "",
        "| Variant | Best Greedy Update | Best Greedy Success | Best Sampled Update | Best Sampled Success | Greedy Checkpoint Exists |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    for variant, variant_rows in sorted(by_variant.items()):
        greedy_rows = [row for row in variant_rows if row["mode"] == "greedy"]
        sampled_rows = [row for row in variant_rows if row["mode"] != "greedy"]
        best_greedy = max(greedy_rows, key=lambda row: row.get("eval_success_rate", 0.0))
        best_sampled = max(sampled_rows, key=lambda row: row.get("eval_success_rate", 0.0)) if sampled_rows else best_greedy
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    str(best_greedy["update"]),
                    _format_float(best_greedy.get("eval_success_rate")),
                    str(best_sampled["update"]),
                    _format_float(best_sampled.get("eval_success_rate")),
                    "yes" if float(best_greedy.get("eval_success_rate", 0.0)) > 0.0 else "no",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    for variant, variant_rows in sorted(by_variant.items()):
        greedy_rows = [row for row in variant_rows if row["mode"] == "greedy"]
        sampled_rows = [row for row in variant_rows if row["mode"] != "greedy"]
        best_greedy = max(greedy_rows, key=lambda row: row.get("eval_success_rate", 0.0))
        best_sampled = max(sampled_rows, key=lambda row: row.get("eval_success_rate", 0.0)) if sampled_rows else best_greedy
        greedy_at_sampled = next(
            row for row in greedy_rows if int(row["update"]) == int(best_sampled["update"])
        )
        if float(best_greedy.get("eval_success_rate", 0.0)) > 0.0:
            lines.append(
                f"- `{variant}` does show a nonzero greedy checkpoint at update `{best_greedy['update']}` "
                f"with greedy success `{_format_float(best_greedy.get('eval_success_rate'))}`."
            )
        else:
            lines.append(
                f"- `{variant}` never shows a nonzero greedy checkpoint in the archived series, "
                f"even though its best sampled checkpoint at update `{best_sampled['update']}` reaches "
                f"sampled success `{_format_float(best_sampled.get('eval_success_rate'))}`."
            )
        lines.append(
            f"- At `{variant}`'s best sampled checkpoint, the corresponding greedy success is "
            f"`{_format_float(greedy_at_sampled.get('eval_success_rate'))}`, eval max-prob is "
            f"`{_format_float(greedy_at_sampled.get('eval/action_max_prob'))}`, and eval margin is "
            f"`{_format_float(greedy_at_sampled.get('eval/action_logit_margin'))}`."
        )
    lines.extend(
        [
            "",
            "## Checkpoint Table",
            "",
            "| Variant | Update | Mode | Eval Success | Eval Return | Eval Max Prob | Eval Margin | Eval Greedy Match | Train Return | Train Success | Throughput | Checkpoint |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["variant"]), int(item["update"]), str(item["mode"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    str(row["update"]),
                    str(row["mode"]),
                    _format_float(row.get("eval_success_rate")),
                    _format_float(row.get("eval_return")),
                    _format_float(row.get("eval/action_max_prob")),
                    _format_float(row.get("eval/action_logit_margin")),
                    _format_float(row.get("eval/action_greedy_match")),
                    _format_float(row.get("train/episode_return")),
                    _format_float(row.get("train/success_rate")),
                    _format_float(row.get("throughput_fps")),
                    f"`{Path(str(row['checkpoint'])).name}`",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint series for DoorKey greedy-recovery runs.")
    parser.add_argument("paths", nargs="+", help="Run directories or parents containing run directories")
    parser.add_argument("--episodes", type=int, default=32)
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
    configure_logging(ctx.is_main_process)
    rows: list[dict[str, Any]] = []
    try:
        for run_dir in run_dirs:
            config = load_config(run_dir / "resolved_config.yaml")
            config.system.device = args.device
            config.logging.tensorboard = False
            set_seed(config.seed + ctx.rank, deterministic=config.system.deterministic)
            train_metrics_by_update = _load_train_metrics_by_update(run_dir)
            envs = make_vector_env(config.env, seed=config.seed, world_rank=ctx.rank)
            model = build_model(config.model, envs.single_observation_space, envs.single_action_space).to(ctx.device)
            envs.close()
            for update, checkpoint_path in _checkpoint_rows(run_dir):
                checkpoint = torch.load(checkpoint_path, map_location=ctx.device, weights_only=False)
                model.load_state_dict(checkpoint["model"])
                train_metrics = train_metrics_by_update.get(update, {})
                for mode_name, greedy, temperature in DEFAULT_MODES:
                    diagnostics = collect_policy_diagnostics(
                        config=config,
                        model=model,
                        ctx=ctx,
                        episodes=args.episodes,
                        greedy=greedy,
                        temperature=temperature,
                        trace_limit=0,
                    )
                    rows.append(
                        {
                            "run_dir": str(run_dir),
                            "checkpoint": str(checkpoint_path),
                            "variant": config.model.variant,
                            "env_id": config.env.env_id,
                            "update": update,
                            "mode": mode_name,
                            "greedy": greedy,
                            "temperature": temperature,
                            **train_metrics,
                            **diagnostics.metrics,
                        }
                    )
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
