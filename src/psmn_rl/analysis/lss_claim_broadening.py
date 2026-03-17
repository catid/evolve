from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_claim_consolidation import (
    _all_seeds,
    _group,
    _lane_seeds,
    _read_csv_rows,
    _summary_stats,
    _with_lane,
)
from psmn_rl.analysis.lss_claim_hardening import _variant_targets
from psmn_rl.analysis.lss_robustness import (
    _best_sampled,
    _evaluate_targets,
    _format_float,
    _greedy_row,
    _maybe_read_command,
    _write_main_process_report,
)
from psmn_rl.utils.io import get_git_commit, get_git_dirty


def _rows_for_lanes(csv_paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane, path in csv_paths:
        rows.extend(_read_csv_rows(path, lane))
    return rows


def _build_reproduction_note(rows: list[dict[str, Any]], episodes: int) -> str:
    grouped = _group(rows)
    labels = [
        ("flat_dense", "flat_dense"),
        ("recovered_token_dense", "recovered token_dense"),
        ("kl_lss_token_dense", "KL learner-state token_dense"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lines = [
        "# Claim-Broadening Reproduction Note",
        "",
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
            "- This note locks the current DoorKey teacher-guided baseline to the external 64-episode decision path before the missing-control, broader-causality, and extra fresh-seed work.",
            "- Forward work in this phase assumes the original and fresh matched DoorKey lanes stay aligned with these already-published artifacts.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_single_expert_report(rows: list[dict[str, Any]], episodes: int) -> str:
    grouped = _group(rows)
    labels = [
        ("flat_dense", "flat_dense"),
        ("recovered_token_dense", "recovered token_dense"),
        ("kl_lss_token_dense", "KL learner-state token_dense"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_single_expert", "KL learner-state single_expert"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lanes = sorted({lane for lane, _seed, _label in grouped})
    lines = [
        "# Matched Teacher-Guided Single-Expert DoorKey Control Report",
        "",
        f"- external evaluation episodes per mode: `{episodes}`",
        "",
        "| Lane | Seed | flat_dense | recovered token_dense | KL learner-state token_dense | baseline PPO SARE | KL learner-state single_expert | KL learner-state SARE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
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
            "| Variant | Seeds Covered | Mean Greedy Success | Min Greedy Success | Max Greedy Success | Complete-Seed Failures |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, display in labels[1:]:
        stats = _summary_stats(grouped, label)
        seed_count = len({seed for _lane, seed, row_label in grouped if row_label == label})
        lines.append(
            f"| {display} | `{seed_count}` | `{stats['mean']:.4f}` | `{stats['min']:.4f}` | `{stats['max']:.4f}` | `{int(stats['zero_failures'])}` |"
        )
    sare_mean = _summary_stats(grouped, "kl_lss_sare")["mean"]
    token_mean = _summary_stats(grouped, "kl_lss_token_dense")["mean"]
    single_mean = _summary_stats(grouped, "kl_lss_single_expert")["mean"]
    lines.extend(["", "## Interpretation", ""])
    if sare_mean > max(token_mean, single_mean):
        lines.append("- On the matched missing-control lane, KL learner-state SARE still stays ahead of both the matched token_dense and matched single_expert teacher-guided controls.")
    elif single_mean >= sare_mean:
        lines.append("- The missing matched single_expert control closes or erases the apparent SARE edge, so the strengthened DoorKey claim should stay method-first rather than routing-first.")
    else:
        lines.append("- Teacher-guided structured students all improve, but the multi-expert SARE edge over single_expert stays narrow rather than decisive.")
    if "fresh" not in lanes:
        lines.append("- This fairness fill-in currently covers the original DoorKey lane only; fresh-lane single_expert controls were not required for the bounded acceptance bar.")
    return "\n".join(lines) + "\n"


def _build_additional_fresh_report(rows: list[dict[str, Any]], episodes: int) -> str:
    grouped = _group(rows)
    lane = next(iter({row["lane"] for row in rows}), "fresh_extra")
    labels = [
        ("flat_dense", "flat_dense"),
        ("recovered_token_dense", "recovered token_dense"),
        ("kl_lss_token_dense", "KL learner-state token_dense"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lines = [
        "# Additional Fresh DoorKey Seed Block Report",
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

    mean_token = _summary_stats(grouped, "kl_lss_token_dense", lane)["mean"]
    mean_sare = _summary_stats(grouped, "kl_lss_sare", lane)["mean"]
    zero_failures = _summary_stats(grouped, "kl_lss_sare", lane)["zero_failures"]
    lines.extend(
        [
            "",
            "## Mean Greedy Success",
            "",
            "| Variant | Mean Greedy Success |",
            "| --- | ---: |",
            f"| recovered token_dense | `{_summary_stats(grouped, 'recovered_token_dense', lane)['mean']:.4f}` |",
            f"| KL learner-state token_dense | `{mean_token:.4f}` |",
            f"| baseline PPO SARE | `{_summary_stats(grouped, 'baseline_sare', lane)['mean']:.4f}` |",
            f"| KL learner-state SARE | `{mean_sare:.4f}` |",
            "",
            "## Interpretation",
            "",
        ]
    )
    if mean_sare > mean_token and zero_failures <= 0.0:
        lines.append("- On the additional fresh matched DoorKey block, KL learner-state SARE still stays ahead of the matched teacher-guided token_dense control without a complete-seed routed failure.")
    elif mean_sare >= mean_token:
        lines.append("- On the additional fresh matched DoorKey block, KL learner-state SARE remains competitive with matched token_dense, but the edge is narrower than the current six-seed picture.")
    else:
        lines.append("- On the additional fresh matched DoorKey block, matched teacher-guided token_dense catches or beats KL learner-state SARE, so the routed edge weakens.")
    return "\n".join(lines) + "\n"


def _build_expanded_combined_report(rows: list[dict[str, Any]], episodes: int) -> str:
    grouped = _group(rows)
    labels = [
        ("flat_dense", "flat_dense"),
        ("recovered_token_dense", "recovered token_dense"),
        ("kl_lss_token_dense", "KL learner-state token_dense"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_single_expert", "KL learner-state single_expert"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lines = [
        "# Expanded Combined DoorKey Claim Report",
        "",
        f"- external evaluation episodes per mode: `{episodes}`",
        "",
        "| Lane | Seed | flat_dense | recovered token_dense | KL learner-state token_dense | baseline PPO SARE | KL learner-state single_expert | KL learner-state SARE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane, seed in _all_seeds(grouped):
        values = [lane, str(seed)]
        for label, _display in labels:
            if (lane, seed, label) in grouped:
                values.append(_format_float(_greedy_row(grouped[(lane, seed, label)]).get("eval_success_rate")))
            else:
                values.append("-")
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Variant | Seeds Covered | Mean Greedy Success | Min Greedy Success | Max Greedy Success | Complete-Seed Failures |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, display in labels:
        stats = _summary_stats(grouped, label)
        seed_count = len({seed for _lane, seed, row_label in grouped if row_label == label})
        lines.append(
            f"| {display} | `{seed_count}` | `{stats['mean']:.4f}` | `{stats['min']:.4f}` | `{stats['max']:.4f}` | `{int(stats['zero_failures'])}` |"
        )
    sare_mean = _summary_stats(grouped, "kl_lss_sare")["mean"]
    token_mean = _summary_stats(grouped, "kl_lss_token_dense")["mean"]
    single_mean = _summary_stats(grouped, "kl_lss_single_expert")["mean"]
    sare_zero = _summary_stats(grouped, "kl_lss_sare")["zero_failures"]
    single_lane = "original" if any(row_label == "kl_lss_single_expert" and lane == "original" for lane, _seed, row_label in grouped) else None
    single_lane_sare_mean = _summary_stats(grouped, "kl_lss_sare", single_lane)["mean"] if single_lane is not None else 0.0
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        f"- Across the expanded DoorKey picture, KL learner-state SARE mean greedy success is `{sare_mean:.4f}` versus `{token_mean:.4f}` for matched KL learner-state token_dense."
    )
    if single_mean > 0.0:
        lines.append(
            f"- On the matched missing-control slice, KL learner-state SARE mean greedy success is `{single_lane_sare_mean:.4f}` versus `{single_mean:.4f}` for KL learner-state single_expert."
        )
    if sare_mean > max(token_mean, single_mean) and sare_zero <= 0.0:
        lines.append("- The expanded DoorKey picture broadens the claim within scope: KL learner-state SARE stays ahead of the matched structured controls without any complete-seed routed failure.")
    elif sare_mean > token_mean:
        lines.append("- The expanded DoorKey picture still favors KL learner-state SARE over matched token_dense, but the missing single_expert control keeps the claim narrower than a clean multi-expert edge.")
    else:
        lines.append("- The expanded DoorKey picture weakens the routed edge: matched structured controls catch up enough that the claim should stay method-first rather than routing-first.")
    return "\n".join(lines) + "\n"


def reproduction_note(args: argparse.Namespace) -> None:
    rows = _rows_for_lanes(
        [
            ("original", Path(args.original_csv)),
            ("fresh", Path(args.fresh_csv)),
        ]
    )
    _write_main_process_report(rows, args.output, args.csv, _build_reproduction_note(rows, args.episodes))


def single_expert_matched_control_report(args: argparse.Namespace) -> None:
    rows = _rows_for_lanes([("original", Path(args.original_csv))])
    targets = _with_lane(
        _variant_targets(Path(args.single_expert_root), {"single_expert": "kl_lss_single_expert"}),
        "original",
    )
    if args.fresh_csv and args.fresh_single_expert_root:
        rows.extend(_rows_for_lanes([("fresh", Path(args.fresh_csv))]))
        targets.extend(
            _with_lane(
                _variant_targets(Path(args.fresh_single_expert_root), {"single_expert": "kl_lss_single_expert"}),
                "fresh",
            )
        )
    rows.extend(_evaluate_targets(targets, args.device, args.episodes))
    keep = {
        "flat_dense",
        "recovered_token_dense",
        "kl_lss_token_dense",
        "baseline_sare",
        "kl_lss_single_expert",
        "kl_lss_sare",
    }
    rows = [row for row in rows if row["label"] in keep]
    _write_main_process_report(rows, args.output, args.csv, _build_single_expert_report(rows, args.episodes))


def additional_fresh_seed_block_report(args: argparse.Namespace) -> None:
    rows = _rows_for_lanes([(args.lane, Path(args.baseline_csv))])
    token_targets = _with_lane(
        _variant_targets(Path(args.token_root), {"token_dense": "kl_lss_token_dense"}),
        args.lane,
    )
    rows.extend(_evaluate_targets(token_targets, args.device, args.episodes))
    keep = {"flat_dense", "recovered_token_dense", "baseline_sare", "kl_lss_token_dense", "kl_lss_sare"}
    rows = [row for row in rows if row["label"] in keep]
    _write_main_process_report(rows, args.output, args.csv, _build_additional_fresh_report(rows, args.episodes))


def expanded_combined_doorkey_report(args: argparse.Namespace) -> None:
    rows = _rows_for_lanes(
        [
            ("original", Path(args.original_csv)),
            ("fresh", Path(args.fresh_csv)),
            (args.extra_lane, Path(args.extra_csv)),
        ]
    )
    if args.single_expert_csv:
        rows.extend(_read_csv_rows(Path(args.single_expert_csv), "original"))
    keep = {
        "flat_dense",
        "recovered_token_dense",
        "kl_lss_token_dense",
        "baseline_sare",
        "kl_lss_single_expert",
        "kl_lss_sare",
    }
    rows = [row for row in rows if row["label"] in keep]
    _write_main_process_report(rows, args.output, args.csv, _build_expanded_combined_report(rows, args.episodes))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze DoorKey claim-broadening experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reproduction = subparsers.add_parser("reproduction-note")
    reproduction.add_argument("--original-csv", required=True)
    reproduction.add_argument("--fresh-csv", required=True)
    reproduction.add_argument("--episodes", type=int, default=64)
    reproduction.add_argument("--output", required=True)
    reproduction.add_argument("--csv", default=None)

    single_expert = subparsers.add_parser("single-expert-matched-control-report")
    single_expert.add_argument("--original-csv", required=True)
    single_expert.add_argument("--single-expert-root", required=True)
    single_expert.add_argument("--episodes", type=int, default=64)
    single_expert.add_argument("--device", default="auto")
    single_expert.add_argument("--output", required=True)
    single_expert.add_argument("--csv", default=None)
    single_expert.add_argument("--fresh-csv", default=None)
    single_expert.add_argument("--fresh-single-expert-root", default=None)

    fresh_block = subparsers.add_parser("additional-fresh-seed-block-report")
    fresh_block.add_argument("--baseline-csv", required=True)
    fresh_block.add_argument("--token-root", required=True)
    fresh_block.add_argument("--episodes", type=int, default=64)
    fresh_block.add_argument("--device", default="auto")
    fresh_block.add_argument("--output", required=True)
    fresh_block.add_argument("--csv", default=None)
    fresh_block.add_argument("--lane", default="fresh_extra")

    combined = subparsers.add_parser("expanded-combined-doorkey-report")
    combined.add_argument("--original-csv", required=True)
    combined.add_argument("--fresh-csv", required=True)
    combined.add_argument("--extra-csv", required=True)
    combined.add_argument("--episodes", type=int, default=64)
    combined.add_argument("--output", required=True)
    combined.add_argument("--csv", default=None)
    combined.add_argument("--single-expert-csv", default=None)
    combined.add_argument("--extra-lane", default="fresh_extra")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "reproduction-note":
        reproduction_note(args)
        return
    if args.command == "single-expert-matched-control-report":
        single_expert_matched_control_report(args)
        return
    if args.command == "additional-fresh-seed-block-report":
        additional_fresh_seed_block_report(args)
        return
    expanded_combined_doorkey_report(args)


if __name__ == "__main__":
    main()
