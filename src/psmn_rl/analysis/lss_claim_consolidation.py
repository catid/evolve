from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_claim_hardening import _filter_baselines, _variant_targets
from psmn_rl.analysis.lss_robustness import (
    EvalTarget,
    _best_sampled,
    _evaluate_targets,
    _format_float,
    _greedy_row,
    _maybe_read_command,
    _write_main_process_report,
)
from psmn_rl.utils.io import get_git_commit, get_git_dirty


def _with_lane(targets: list[EvalTarget], lane: str) -> list[EvalTarget]:
    updated: list[EvalTarget] = []
    for target in targets:
        metadata = dict(target.metadata or {})
        metadata["lane"] = lane
        updated.append(
            EvalTarget(
                seed=target.seed,
                label=target.label,
                variant=target.variant,
                config_path=target.config_path,
                checkpoint_path=target.checkpoint_path,
                run_dir=target.run_dir,
                method=target.method,
                stage=lane,
                round_index=target.round_index,
                round_diagnostics=target.round_diagnostics,
                metadata=metadata,
                command_path=target.command_path,
            )
        )
    return updated


def _read_csv_rows(path: Path, lane: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for row in rows:
        row["lane"] = lane
    return rows


def _group(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("lane", row.get("stage", "unknown"))), int(row["seed"]), str(row["label"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def _lane_seeds(grouped: dict[tuple[str, int, str], list[dict[str, Any]]], lane: str) -> list[int]:
    return sorted({seed for row_lane, seed, _label in grouped if row_lane == lane})


def _all_seeds(grouped: dict[tuple[str, int, str], list[dict[str, Any]]]) -> list[tuple[str, int]]:
    return sorted({(lane, seed) for lane, seed, _label in grouped})


def _summary_stats(grouped: dict[tuple[str, int, str], list[dict[str, Any]]], label: str, lane: str | None = None) -> dict[str, float]:
    values: list[float] = []
    for (row_lane, _seed, row_label), mode_rows in grouped.items():
        if row_label != label:
            continue
        if lane is not None and row_lane != lane:
            continue
        values.append(float(_greedy_row(mode_rows).get("eval_success_rate", 0.0)))
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "zero_failures": 0.0}
    zero_failures = sum(1 for value in values if value <= 0.0)
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "zero_failures": float(zero_failures),
    }


def _build_reproduction_note(
    rows: list[dict[str, Any]],
    original_baseline_root: Path,
    original_improved_root: Path,
    fresh_baseline_root: Path,
    fresh_improved_root: Path,
    episodes: int,
) -> str:
    grouped = _group(rows)
    labels = [
        ("flat_dense", "flat_dense"),
        ("recovered_token_dense", "recovered token_dense"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lines = [
        "# Claim-Consolidation Reproduction Note",
        "",
        f"- original baseline root: `{original_baseline_root}`",
        f"- original improved root: `{original_improved_root}`",
        f"- fresh baseline root: `{fresh_baseline_root}`",
        f"- fresh improved root: `{fresh_improved_root}`",
        f"- external evaluation episodes per mode: `{episodes}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Per-Seed Results",
        "",
        "| Lane | Seed | Variant | Greedy Success | Best Sampled Success | Best Sampled Mode | Config | Checkpoint | Command |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for lane, seed in _all_seeds(grouped):
        for label, display in labels:
            mode_rows = grouped[(lane, seed, label)]
            greedy = _greedy_row(mode_rows)
            best_mode, best_success = _best_sampled(mode_rows)
            lines.append(
                "| "
                + " | ".join(
                    [
                        lane,
                        str(seed),
                        display,
                        _format_float(greedy.get("eval_success_rate")),
                        _format_float(best_success),
                        best_mode,
                        f"`{greedy['config_path']}`",
                        f"`{greedy['checkpoint_path']}`",
                        f"`{_maybe_read_command(Path(greedy['run_dir']) / 'command.txt') or '-'}`",
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This note re-evaluates both the original and fresh DoorKey teacher-guided SARE lanes on the external 64-episode decision path before the fresh matched-control and causal-route probes.",
            "- Forward work in this phase is only valid if both lanes remain close to the already-published claim-hardening artifacts.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_fresh_matched_control_report(rows: list[dict[str, Any]], episodes: int) -> str:
    grouped = _group(rows)
    lane = next(iter({row["lane"] for row in rows}), "fresh")
    labels = [
        ("flat_dense", "flat_dense"),
        ("recovered_token_dense", "recovered token_dense"),
        ("kl_lss_token_dense", "KL learner-state token_dense"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lines = [
        "# Fresh Matched Teacher-Guided DoorKey Control Report",
        "",
        f"- lane: `{lane}`",
        f"- external evaluation episodes per mode: `{episodes}`",
        "",
        "| Seed | flat_dense | recovered token_dense | KL learner-state token_dense | baseline PPO SARE | KL learner-state SARE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in _lane_seeds(grouped, lane):
        values = [str(seed)]
        for label, _display in labels:
            values.append(_format_float(_greedy_row(grouped[(lane, seed, label)]).get("eval_success_rate")))
        lines.append("| " + " | ".join(values) + " |")

    mean_recovered = _summary_stats(grouped, "recovered_token_dense", lane)["mean"]
    mean_lss_token = _summary_stats(grouped, "kl_lss_token_dense", lane)["mean"]
    mean_baseline = _summary_stats(grouped, "baseline_sare", lane)["mean"]
    mean_lss_sare = _summary_stats(grouped, "kl_lss_sare", lane)["mean"]
    zero_failures = _summary_stats(grouped, "kl_lss_sare", lane)["zero_failures"]
    lines.extend(
        [
            "",
            "## Mean Greedy Success",
            "",
            "| Variant | Mean Greedy Success |",
            "| --- | ---: |",
            f"| recovered token_dense | `{mean_recovered:.4f}` |",
            f"| KL learner-state token_dense | `{mean_lss_token:.4f}` |",
            f"| baseline PPO SARE | `{mean_baseline:.4f}` |",
            f"| KL learner-state SARE | `{mean_lss_sare:.4f}` |",
            "",
            "## Interpretation",
            "",
        ]
    )
    if mean_lss_sare > mean_lss_token and zero_failures <= 0.0:
        lines.append("- On the fresh matched teacher-guided lane, KL learner-state SARE still holds a mean greedy-success edge over KL learner-state token_dense and avoids complete-seed failure.")
    elif mean_lss_sare >= mean_lss_token:
        lines.append("- On the fresh matched teacher-guided lane, KL learner-state SARE stays competitive with KL learner-state token_dense, but complete-seed failure keeps the routed edge narrow.")
    else:
        lines.append("- On the fresh matched teacher-guided lane, KL learner-state token_dense matches or beats KL learner-state SARE, so the routed edge does not harden beyond a method-first claim.")
    return "\n".join(lines) + "\n"


def _build_combined_doorkey_report(rows: list[dict[str, Any]], episodes: int) -> str:
    grouped = _group(rows)
    labels = [
        ("flat_dense", "flat_dense"),
        ("recovered_token_dense", "recovered token_dense"),
        ("kl_lss_token_dense", "KL learner-state token_dense"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lanes = sorted({lane for lane, _seed, _label in grouped})
    lines = [
        "# Combined DoorKey Claim Report",
        "",
        f"- external evaluation episodes per mode: `{episodes}`",
        "",
        "| Lane | Seed | flat_dense | recovered token_dense | KL learner-state token_dense | baseline PPO SARE | KL learner-state SARE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane, seed in _all_seeds(grouped):
        values = [lane, str(seed)]
        for label, _display in labels:
            values.append(_format_float(_greedy_row(grouped[(lane, seed, label)]).get("eval_success_rate")))
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Variant | Mean Greedy Success | Min Greedy Success | Max Greedy Success | Complete-Seed Failures |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, display in labels:
        stats = _summary_stats(grouped, label)
        lines.append(
            f"| {display} | `{stats['mean']:.4f}` | `{stats['min']:.4f}` | `{stats['max']:.4f}` | `{int(stats['zero_failures'])}` |"
        )

    original_sare = _summary_stats(grouped, "kl_lss_sare", "original")["mean"]
    fresh_sare = _summary_stats(grouped, "kl_lss_sare", "fresh")["mean"]
    original_token = _summary_stats(grouped, "kl_lss_token_dense", "original")["mean"]
    fresh_token = _summary_stats(grouped, "kl_lss_token_dense", "fresh")["mean"]
    combined_sare = _summary_stats(grouped, "kl_lss_sare")["mean"]
    combined_token = _summary_stats(grouped, "kl_lss_token_dense")["mean"]
    sare_zero_failures = _summary_stats(grouped, "kl_lss_sare")["zero_failures"]
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        f"- On the original matched lane, KL learner-state SARE mean greedy success is `{original_sare:.4f}` versus `{original_token:.4f}` for KL learner-state token_dense."
    )
    lines.append(
        f"- On the fresh matched lane, KL learner-state SARE mean greedy success is `{fresh_sare:.4f}` versus `{fresh_token:.4f}` for KL learner-state token_dense."
    )
    if combined_sare > combined_token and sare_zero_failures <= 0.0:
        lines.append("- The combined six-seed DoorKey picture strengthens the routed edge: KL learner-state SARE stays ahead of the matched teacher-guided token_dense control without any complete-seed greedy failures.")
    elif combined_sare >= combined_token:
        lines.append("- The combined DoorKey picture still favors KL learner-state SARE slightly, but the edge remains narrow rather than decisive.")
    else:
        lines.append("- The combined DoorKey picture is method-first rather than routing-first: matched teacher-guided token_dense is at least as strong as KL learner-state SARE.")
    if "fresh" not in lanes or "original" not in lanes:
        lines.append("- Warning: one of the expected lanes is missing from this combined report.")
    return "\n".join(lines) + "\n"


def reproduction_note(args: argparse.Namespace) -> None:
    if args.original_csv and args.fresh_csv:
        rows = _read_csv_rows(Path(args.original_csv), "original") + _read_csv_rows(Path(args.fresh_csv), "fresh")
        _write_main_process_report(
            rows,
            args.output,
            args.csv,
            _build_reproduction_note(
                rows,
                Path(args.original_baseline_root),
                Path(args.original_improved_root),
                Path(args.fresh_baseline_root),
                Path(args.fresh_improved_root),
                args.episodes,
            ),
        )
        return
    original_baselines = _with_lane(
        _filter_baselines(Path(args.original_baseline_root), {"flat_dense", "recovered_token_dense", "baseline_sare"}),
        "original",
    )
    original_improved = _with_lane(_variant_targets(Path(args.original_improved_root), {"sare": "kl_lss_sare"}), "original")
    fresh_baselines = _with_lane(
        _filter_baselines(Path(args.fresh_baseline_root), {"flat_dense", "recovered_token_dense", "baseline_sare"}),
        "fresh",
    )
    fresh_improved = _with_lane(_variant_targets(Path(args.fresh_improved_root), {"sare": "kl_lss_sare"}), "fresh")
    rows = _evaluate_targets(original_baselines + original_improved + fresh_baselines + fresh_improved, args.device, args.episodes)
    _write_main_process_report(
        rows,
        args.output,
        args.csv,
        _build_reproduction_note(
            rows,
            Path(args.original_baseline_root),
            Path(args.original_improved_root),
            Path(args.fresh_baseline_root),
            Path(args.fresh_improved_root),
            args.episodes,
        ),
    )


def fresh_matched_control_report(args: argparse.Namespace) -> None:
    if args.baseline_csv:
        baseline_rows = _read_csv_rows(Path(args.baseline_csv), "fresh")
        keep = {"flat_dense", "recovered_token_dense", "baseline_sare", "kl_lss_sare"}
        baseline_rows = [row for row in baseline_rows if row["label"] in keep]
        token_targets = _with_lane(_variant_targets(Path(args.token_root), {"token_dense": "kl_lss_token_dense"}), "fresh")
        token_rows = _evaluate_targets(token_targets, args.device, args.episodes)
        rows = baseline_rows + token_rows
        _write_main_process_report(rows, args.output, args.csv, _build_fresh_matched_control_report(rows, args.episodes))
        return
    baseline_targets = _with_lane(
        _filter_baselines(Path(args.baseline_root), {"flat_dense", "recovered_token_dense", "baseline_sare"}),
        "fresh",
    )
    sare_targets = _with_lane(_variant_targets(Path(args.sare_root), {"sare": "kl_lss_sare"}), "fresh")
    token_targets = _with_lane(_variant_targets(Path(args.token_root), {"token_dense": "kl_lss_token_dense"}), "fresh")
    rows = _evaluate_targets(baseline_targets + sare_targets + token_targets, args.device, args.episodes)
    _write_main_process_report(rows, args.output, args.csv, _build_fresh_matched_control_report(rows, args.episodes))


def combined_doorkey_report(args: argparse.Namespace) -> None:
    if args.original_csv and args.fresh_csv:
        rows = _read_csv_rows(Path(args.original_csv), "original") + _read_csv_rows(Path(args.fresh_csv), "fresh")
        _write_main_process_report(rows, args.output, args.csv, _build_combined_doorkey_report(rows, args.episodes))
        return
    original_targets = _with_lane(
        _filter_baselines(Path(args.original_baseline_root), {"flat_dense", "recovered_token_dense", "baseline_sare"}),
        "original",
    ) + _with_lane(_variant_targets(Path(args.original_sare_root), {"sare": "kl_lss_sare"}), "original") + _with_lane(
        _variant_targets(Path(args.original_token_root), {"token_dense": "kl_lss_token_dense"}),
        "original",
    )
    fresh_targets = _with_lane(
        _filter_baselines(Path(args.fresh_baseline_root), {"flat_dense", "recovered_token_dense", "baseline_sare"}),
        "fresh",
    ) + _with_lane(_variant_targets(Path(args.fresh_sare_root), {"sare": "kl_lss_sare"}), "fresh") + _with_lane(
        _variant_targets(Path(args.fresh_token_root), {"token_dense": "kl_lss_token_dense"}),
        "fresh",
    )
    rows = _evaluate_targets(original_targets + fresh_targets, args.device, args.episodes)
    _write_main_process_report(rows, args.output, args.csv, _build_combined_doorkey_report(rows, args.episodes))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze DoorKey claim-consolidation experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reproduction = subparsers.add_parser("reproduction-note")
    reproduction.add_argument("--original-baseline-root", required=True)
    reproduction.add_argument("--original-improved-root", required=True)
    reproduction.add_argument("--fresh-baseline-root", required=True)
    reproduction.add_argument("--fresh-improved-root", required=True)
    reproduction.add_argument("--episodes", type=int, default=64)
    reproduction.add_argument("--device", default="auto")
    reproduction.add_argument("--output", required=True)
    reproduction.add_argument("--csv", default=None)
    reproduction.add_argument("--original-csv", default=None)
    reproduction.add_argument("--fresh-csv", default=None)

    fresh = subparsers.add_parser("fresh-matched-control-report")
    fresh.add_argument("--baseline-root", required=True)
    fresh.add_argument("--sare-root", required=True)
    fresh.add_argument("--token-root", required=True)
    fresh.add_argument("--episodes", type=int, default=64)
    fresh.add_argument("--device", default="auto")
    fresh.add_argument("--output", required=True)
    fresh.add_argument("--csv", default=None)
    fresh.add_argument("--baseline-csv", default=None)

    combined = subparsers.add_parser("combined-doorkey-report")
    combined.add_argument("--original-baseline-root", required=True)
    combined.add_argument("--original-sare-root", required=True)
    combined.add_argument("--original-token-root", required=True)
    combined.add_argument("--fresh-baseline-root", required=True)
    combined.add_argument("--fresh-sare-root", required=True)
    combined.add_argument("--fresh-token-root", required=True)
    combined.add_argument("--episodes", type=int, default=64)
    combined.add_argument("--device", default="auto")
    combined.add_argument("--output", required=True)
    combined.add_argument("--csv", default=None)
    combined.add_argument("--original-csv", default=None)
    combined.add_argument("--fresh-csv", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "reproduction-note":
        reproduction_note(args)
        return
    if args.command == "fresh-matched-control-report":
        fresh_matched_control_report(args)
        return
    combined_doorkey_report(args)


if __name__ == "__main__":
    main()
