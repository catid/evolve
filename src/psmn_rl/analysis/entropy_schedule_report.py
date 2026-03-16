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


def _load_final_train_metrics(run_dir: Path) -> dict[str, float]:
    metrics_path = run_dir / "metrics.jsonl"
    final: dict[str, float] = {}
    if not metrics_path.exists():
        return final
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("type") != "scalar":
            continue
        final = {
            key: float(value)
            for key, value in row.items()
            if key not in {"type", "step"} and isinstance(value, (int, float))
        }
    return final


def _schedule_label(config: Any) -> str:
    ppo = config.ppo
    initial = f"{ppo.ent_coef:g}"
    final = initial if ppo.ent_coef_final is None else f"{ppo.ent_coef_final:g}"
    if ppo.ent_schedule == "constant" or ppo.ent_coef_final is None:
        return f"constant:{initial}"
    if ppo.ent_schedule == "linear":
        return f"linear:{initial}->{final}"
    if ppo.ent_schedule == "late_linear":
        return f"late_linear:{initial}->{final}@{ppo.ent_schedule_start_fraction:g}"
    if ppo.ent_schedule == "step":
        return f"step:{initial}->{final}@{ppo.ent_schedule_start_fraction:g}"
    return f"{ppo.ent_schedule}:{initial}->{final}"


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _run_sort_key(run_rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    greedy_row = next(row for row in run_rows if row["mode"] == "greedy")
    best_sampled = max(
        [row for row in run_rows if row["mode"] != "greedy"],
        key=lambda row: row.get("eval_success_rate", 0.0),
    )
    return (
        float(greedy_row.get("eval_success_rate", 0.0)),
        float(best_sampled.get("eval_success_rate", 0.0)),
        float(greedy_row.get("eval/action_logit_margin", 0.0)),
    )


def _build_report(rows: list[dict[str, Any]], episodes: int) -> str:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(str(row["run_dir"]), []).append(row)
    by_variant: dict[str, list[list[dict[str, Any]]]] = {}
    for run_rows in by_run.values():
        by_variant.setdefault(str(run_rows[0]["variant"]), []).append(run_rows)

    lines = [
        "# Entropy Schedule Report",
        "",
        f"- episodes per mode: `{episodes}`",
        f"- run count: `{len(by_run)}`",
        "",
        "## Best Schedules By Variant",
        "",
        "| Variant | Schedule | Greedy Success | Best Sampled Success | Greedy Max Prob | Greedy Margin | Train Return |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, variant_runs in sorted(by_variant.items()):
        best_run = max(variant_runs, key=_run_sort_key)
        greedy_row = next(row for row in best_run if row["mode"] == "greedy")
        best_sampled = max(
            [row for row in best_run if row["mode"] != "greedy"],
            key=lambda row: row.get("eval_success_rate", 0.0),
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    str(greedy_row["schedule"]),
                    _format_float(greedy_row.get("eval_success_rate")),
                    _format_float(best_sampled.get("eval_success_rate")),
                    _format_float(greedy_row.get("eval/action_max_prob")),
                    _format_float(greedy_row.get("eval/action_logit_margin")),
                    _format_float(greedy_row.get("train/episode_return")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Interpretation", ""])
    for variant, variant_runs in sorted(by_variant.items()):
        best_run = max(variant_runs, key=_run_sort_key)
        greedy_row = next(row for row in best_run if row["mode"] == "greedy")
        best_sampled = max(
            [row for row in best_run if row["mode"] != "greedy"],
            key=lambda row: row.get("eval_success_rate", 0.0),
        )
        if float(greedy_row.get("eval_success_rate", 0.0)) > 0.0:
            lines.append(
                f"- `{variant}` shows greedy improvement under `{greedy_row['schedule']}`: "
                f"greedy success `{_format_float(greedy_row.get('eval_success_rate'))}`, "
                f"best sampled success `{_format_float(best_sampled.get('eval_success_rate'))}`."
            )
        else:
            lines.append(
                f"- `{variant}` does not recover a nonzero greedy policy under any tested schedule. "
                f"The best schedule `{greedy_row['schedule']}` still has greedy success "
                f"`{_format_float(greedy_row.get('eval_success_rate'))}` while best sampled success is "
                f"`{_format_float(best_sampled.get('eval_success_rate'))}`."
            )

    lines.extend(
        [
            "",
            "## Schedule Table",
            "",
            "| Variant | Schedule | Mode | Eval Success | Eval Return | Eval Max Prob | Eval Margin | Train Return | Entropy Coef | Throughput | Run |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for run_rows in sorted(by_run.values(), key=lambda rows: (str(rows[0]["variant"]), str(rows[0]["schedule"]))):
        for row in sorted(run_rows, key=lambda item: str(item["mode"])):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["variant"]),
                        str(row["schedule"]),
                        str(row["mode"]),
                        _format_float(row.get("eval_success_rate")),
                        _format_float(row.get("eval_return")),
                        _format_float(row.get("eval/action_max_prob")),
                        _format_float(row.get("eval/action_logit_margin")),
                        _format_float(row.get("train/episode_return")),
                        _format_float(row.get("ent_coef_current")),
                        _format_float(row.get("throughput_fps")),
                        f"`{Path(str(row['run_dir'])).name}`",
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate latest checkpoints for entropy schedule sweeps.")
    parser.add_argument("paths", nargs="+", help="Run directories or parent directories containing run directories")
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
            train_metrics = _load_final_train_metrics(run_dir)
            envs = make_vector_env(config.env, seed=config.seed, world_rank=ctx.rank)
            model = build_model(config.model, envs.single_observation_space, envs.single_action_space).to(ctx.device)
            envs.close()
            checkpoint = torch.load(run_dir / "latest.pt", map_location=ctx.device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            schedule = _schedule_label(config)
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
                        "variant": config.model.variant,
                        "schedule": schedule,
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
