from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SCALAR_TYPE = "scalar"


@dataclass(slots=True)
class RunSummary:
    run_dir: Path
    run_name: str
    env_id: str
    env_name: str
    variant: str
    seed: int | None
    greedy_eval: bool | None
    fully_observed: bool
    final_metrics: dict[str, float]
    best_metrics: dict[str, float]

    def to_row(self) -> dict[str, str | float | int]:
        final_train_return = self.final_metrics.get("train/episode_return", 0.0)
        final_train_success = self.final_metrics.get("train/success_rate", 0.0)
        eval_return = self.final_metrics.get("eval_return", 0.0)
        eval_success = self.final_metrics.get("eval_success_rate", 0.0)
        return {
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "env_name": self.env_name,
            "env_id": self.env_id,
            "variant": self.variant,
            "seed": self.seed if self.seed is not None else "",
            "greedy_eval": "" if self.greedy_eval is None else str(self.greedy_eval).lower(),
            "fully_observed": str(self.fully_observed).lower(),
            "final_train_return": final_train_return,
            "best_train_return": self.best_metrics.get("train/episode_return", 0.0),
            "final_eval_return": eval_return,
            "train_eval_return_gap": final_train_return - eval_return,
            "final_train_success": final_train_success,
            "best_train_success": self.best_metrics.get("train/success_rate", 0.0),
            "final_eval_success": eval_success,
            "train_eval_success_gap": final_train_success - eval_success,
            "throughput_fps": self.final_metrics.get("throughput_fps", 0.0),
            "explained_variance": self.final_metrics.get("explained_variance", 0.0),
            "entropy": self.final_metrics.get("entropy", 0.0),
            "loss": self.final_metrics.get("loss", 0.0),
            "train_episode_length": self.final_metrics.get("train/episode_length", 0.0),
            "eval_episode_length": self.final_metrics.get("eval_episode_length", 0.0),
            "active_compute_proxy": self.final_metrics.get("active_compute_proxy", 0.0),
            "route_entropy": self.final_metrics.get("route_entropy", 0.0),
            "path_entropy": self.final_metrics.get("path_entropy", 0.0),
            "avg_hop_count": self.final_metrics.get("avg_hop_count", 0.0),
            "avg_halting_probability": self.final_metrics.get("avg_halting_probability", 0.0),
            "relational_usage_rate": self.final_metrics.get("relational_usage_rate", 0.0),
            "option_duration": self.final_metrics.get("option_duration", 0.0),
            "option_switch_rate": self.final_metrics.get("option_switch_rate", 0.0),
        }


def _resolve_metrics_paths(inputs: list[str]) -> list[Path]:
    discovered: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_file():
            discovered.append(path)
            continue
        if (path / "metrics.jsonl").exists():
            discovered.append(path / "metrics.jsonl")
            continue
        if path.is_dir():
            discovered.extend(sorted(path.glob("*/metrics.jsonl")))
    unique_paths: dict[Path, None] = {}
    for item in discovered:
        unique_paths[item.resolve()] = None
    return sorted(unique_paths.keys())


def _load_scalar_series(metrics_path: Path) -> list[dict[str, float]]:
    series: list[dict[str, float]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("type") != SCALAR_TYPE:
                continue
            scalar_payload = {
                key: float(value)
                for key, value in payload.items()
                if key not in {"type", "step"}
            }
            scalar_payload["step"] = float(payload["step"])
            series.append(scalar_payload)
    return series


def _best_metrics(series: list[dict[str, float]]) -> dict[str, float]:
    if not series:
        return {}
    keys = {key for item in series for key in item.keys() if key != "step"}
    best: dict[str, float] = {}
    for key in sorted(keys):
        values = [item[key] for item in series if key in item]
        if values:
            best[key] = max(values)
    return best


def _canonical_env_name(env_id: str) -> str:
    env_name = str(env_id)
    for prefix in ("MiniGrid-", "procgen_gym/procgen-"):
        env_name = env_name.removeprefix(prefix)
    env_name = env_name.removesuffix("-v0")
    env_name = env_name.replace("Dynamic-Obstacles-", "dynamic_obstacles_")
    return env_name.replace("-", "_").lower()


def _load_resolved_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text()) or {}


def _infer_variant(run_name: str) -> str:
    for variant in ("flat_dense", "token_dense", "token_gru", "single_expert", "sare", "treg_h", "srw", "por"):
        if run_name.endswith(variant):
            return variant
    return run_name


def load_run_summary(metrics_path: Path) -> RunSummary:
    series = _load_scalar_series(metrics_path)
    final_metrics = series[-1] if series else {}
    best_metrics = _best_metrics(series)
    run_dir = metrics_path.parent
    config = _load_resolved_config(run_dir)
    env_cfg = config.get("env", {})
    env_id = str(env_cfg.get("env_id", run_dir.name))
    fully_observed = bool(env_cfg.get("fully_observed", False))
    env_name = _canonical_env_name(env_id)
    if fully_observed:
        env_name = f"{env_name}_fullobs"
    variant = str(config.get("model", {}).get("variant", _infer_variant(run_dir.name)))
    evaluation_cfg = config.get("evaluation", {})
    seed = config.get("seed")
    return RunSummary(
        run_dir=run_dir,
        run_name=run_dir.name,
        env_id=env_id,
        env_name=env_name,
        variant=variant,
        seed=int(seed) if seed is not None else None,
        greedy_eval=bool(evaluation_cfg["greedy"]) if "greedy" in evaluation_cfg else None,
        fully_observed=fully_observed,
        final_metrics=final_metrics,
        best_metrics=best_metrics,
    )


def _format_number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _aggregate(rows: list[RunSummary]) -> dict[str, float]:
    if not rows:
        return {}
    numeric_keys = {
        key
        for row in rows
        for key, value in row.to_row().items()
        if isinstance(value, float)
    }
    aggregate = {"seeds": float(len(rows))}
    for key in numeric_keys:
        values = [float(row.to_row()[key]) for row in rows]
        aggregate[key] = sum(values) / len(values)
    return aggregate


def build_report(runs: list[RunSummary]) -> str:
    if not runs:
        return "# Run Summary\n\nNo runs found."

    grouped: dict[str, dict[str, list[RunSummary]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        grouped[run.env_name][run.variant].append(run)

    lines = ["# Run Summary", ""]
    lines.append("## Aggregate By Environment / Variant")
    lines.append("")
    for env_name in sorted(grouped):
        lines.append(f"### {env_name}")
        lines.append("")
        lines.append(
            "| variant | seeds | eval_return | eval_success | best_train_return | train_eval_gap | throughput | compute_proxy | route_entropy | hop_count | rel_usage | option_duration |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for variant in sorted(grouped[env_name]):
            aggregate = _aggregate(grouped[env_name][variant])
            lines.append(
                "| {variant} | {seeds:.0f} | {eval_return} | {eval_success} | {best_train_return} | {train_eval_gap} | {throughput} | {compute_proxy} | {route_entropy} | {hop_count} | {rel_usage} | {option_duration} |".format(
                    variant=variant,
                    seeds=aggregate.get("seeds", 0.0),
                    eval_return=_format_number(aggregate.get("final_eval_return", 0.0)),
                    eval_success=_format_number(aggregate.get("final_eval_success", 0.0)),
                    best_train_return=_format_number(aggregate.get("best_train_return", 0.0)),
                    train_eval_gap=_format_number(aggregate.get("train_eval_return_gap", 0.0)),
                    throughput=_format_number(aggregate.get("throughput_fps", 0.0), digits=1),
                    compute_proxy=_format_number(aggregate.get("active_compute_proxy", 0.0)),
                    route_entropy=_format_number(aggregate.get("route_entropy", 0.0)),
                    hop_count=_format_number(aggregate.get("avg_hop_count", 0.0)),
                    rel_usage=_format_number(aggregate.get("relational_usage_rate", 0.0)),
                    option_duration=_format_number(aggregate.get("option_duration", 0.0)),
                )
            )
        lines.append("")

    lines.append("## Per-Run Details")
    lines.append("")
    for env_name in sorted(grouped):
        lines.append(f"### {env_name}")
        lines.append("")
        lines.append(
            "| variant | seed | greedy_eval | fullobs | eval_return | eval_success | final_train_return | best_train_return | train_eval_gap | explained_variance | entropy | throughput | run |"
        )
        lines.append(
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
        )
        env_runs = sorted(grouped[env_name].values(), key=lambda rows: (rows[0].variant, rows[0].seed or -1))
        for rows in env_runs:
            for run in sorted(rows, key=lambda item: (item.variant, item.seed or -1, item.run_name)):
                row = run.to_row()
                lines.append(
                    "| {variant} | {seed} | {greedy_eval} | {fully_observed} | {eval_return} | {eval_success} | {final_train_return} | {best_train_return} | {train_eval_gap} | {explained_variance} | {entropy} | {throughput} | `{run_name}` |".format(
                        variant=row["variant"],
                        seed=row["seed"],
                        greedy_eval=row["greedy_eval"],
                        fully_observed=row["fully_observed"],
                        eval_return=_format_number(float(row["final_eval_return"])),
                        eval_success=_format_number(float(row["final_eval_success"])),
                        final_train_return=_format_number(float(row["final_train_return"])),
                        best_train_return=_format_number(float(row["best_train_return"])),
                        train_eval_gap=_format_number(float(row["train_eval_return_gap"])),
                        explained_variance=_format_number(float(row["explained_variance"])),
                        entropy=_format_number(float(row["entropy"])),
                        throughput=_format_number(float(row["throughput_fps"]), digits=1),
                        run_name=row["run_name"],
                    )
                )
        lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, runs: list[RunSummary]) -> None:
    rows = [run.to_row() for run in runs]
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PSMN RL runs from metrics JSONL files.")
    parser.add_argument("runs", nargs="+", help="Metrics files, run directories, or roots containing per-run folders")
    parser.add_argument("--output", default=None, help="Optional markdown output path")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    metrics_paths = _resolve_metrics_paths(args.runs)
    run_summaries = [load_run_summary(path) for path in metrics_paths]
    report = build_report(run_summaries)
    if args.output is not None:
        Path(args.output).write_text(report)
    if args.csv is not None:
        write_csv(Path(args.csv), run_summaries)
    print(report)


if __name__ == "__main__":
    main()
