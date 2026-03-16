from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_robustness import (
    EvalTarget,
    _baseline_targets,
    _best_sampled,
    _evaluate_targets,
    _format_float,
    _greedy_row,
    _maybe_read_command,
    _sweep_targets,
    _write_main_process_report,
)
from psmn_rl.utils.io import get_git_commit, get_git_dirty


def _variant_targets(root: Path, mapping: dict[str, str]) -> list[EvalTarget]:
    targets: list[EvalTarget] = []
    for target in _sweep_targets([str(root)]):
        label = mapping.get(target.variant)
        if label is None:
            continue
        targets.append(
            EvalTarget(
                seed=target.seed,
                label=label,
                variant=target.variant,
                config_path=target.config_path,
                checkpoint_path=target.checkpoint_path,
                run_dir=target.run_dir,
                method=target.method,
                stage=target.stage,
                command_path=target.command_path,
            )
        )
    return targets


def _filter_baselines(root: Path, keep: set[str]) -> list[EvalTarget]:
    return [target for target in _baseline_targets(root) if target.label in keep]


def _group(rows: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["seed"]), str(row["label"])), []).append(row)
    return grouped


def _mean_greedy(grouped: dict[tuple[int, str], list[dict[str, Any]]], label: str) -> float:
    seeds = sorted({seed for seed, row_label in grouped if row_label == label})
    if not seeds:
        return 0.0
    return sum(float(_greedy_row(grouped[(seed, label)]).get("eval_success_rate", 0.0)) for seed in seeds) / len(seeds)


def _zero_failure(grouped: dict[tuple[int, str], list[dict[str, Any]]], label: str) -> bool:
    seeds = sorted({seed for seed, row_label in grouped if row_label == label})
    return any(float(_greedy_row(grouped[(seed, label)]).get("eval_success_rate", 0.0)) <= 0.0 for seed in seeds)


def _build_reproduction_note(rows: list[dict[str, Any]], baseline_root: Path, improved_root: Path, episodes: int) -> str:
    grouped = _group(rows)
    labels = [
        ("flat_dense", "flat_dense"),
        ("recovered_token_dense", "recovered token_dense"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lines = [
        "# Claim-Hardening Reproduction Note",
        "",
        f"- baseline root: `{baseline_root}`",
        f"- improved root: `{improved_root}`",
        f"- external evaluation episodes per mode: `{episodes}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Per-Seed Results",
        "",
        "| Seed | Variant | Greedy Success | Best Sampled Success | Best Sampled Mode | Config | Checkpoint | Command |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for seed in sorted({key[0] for key in grouped}):
        for label, display in labels:
            mode_rows = grouped[(seed, label)]
            greedy = _greedy_row(mode_rows)
            best_mode, best_success = _best_sampled(mode_rows)
            lines.append(
                "| "
                + " | ".join(
                    [
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
            "- This note re-evaluates the current DoorKey teacher/student baseline on the external 64-episode decision path before any new seed or matched-control extension.",
            "- Forward work in this phase is only valid if the reproduced KL learner-state SARE lane stays close to the published robustness artifacts.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_additional_seed_report(rows: list[dict[str, Any]], episodes: int) -> str:
    grouped = _group(rows)
    labels = [
        ("flat_dense", "flat_dense"),
        ("recovered_token_dense", "recovered token_dense"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lines = [
        "# Additional-Seed DoorKey Report",
        "",
        f"- external evaluation episodes per mode: `{episodes}`",
        "",
        "| Seed | flat_dense | recovered token_dense | baseline PPO SARE | KL learner-state SARE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for seed in sorted({key[0] for key in grouped}):
        values = [str(seed)]
        for label, _display in labels:
            values.append(_format_float(_greedy_row(grouped[(seed, label)]).get("eval_success_rate")))
        lines.append("| " + " | ".join(values) + " |")
    mean_flat = _mean_greedy(grouped, "flat_dense")
    mean_token = _mean_greedy(grouped, "recovered_token_dense")
    mean_baseline = _mean_greedy(grouped, "baseline_sare")
    mean_improved = _mean_greedy(grouped, "kl_lss_sare")
    lines.extend(
        [
            "",
            "## Mean Greedy Success",
            "",
            "| Variant | Mean Greedy Success |",
            "| --- | ---: |",
            f"| flat_dense | `{mean_flat:.4f}` |",
            f"| recovered token_dense | `{mean_token:.4f}` |",
            f"| baseline PPO SARE | `{mean_baseline:.4f}` |",
            f"| KL learner-state SARE | `{mean_improved:.4f}` |",
            "",
            "## Verdict",
            "",
        ]
    )
    no_zero_fail = not _zero_failure(grouped, "kl_lss_sare")
    if mean_improved >= mean_token and no_zero_fail:
        lines.append("- On the additional seed set, KL learner-state SARE stays competitive with recovered token_dense and avoids complete-seed failure.")
    elif mean_improved > mean_baseline:
        lines.append("- On the additional seed set, KL learner-state SARE still beats baseline PPO SARE, but the reopened claim weakens because it either trails recovered token_dense or keeps a complete-seed failure.")
    else:
        lines.append("- On the additional seed set, KL learner-state SARE no longer beats the baseline PPO routed control strongly enough to keep the reopened claim alive.")
    if not no_zero_fail:
        lines.append("- At least one new seed remains a complete greedy failure at `0.0`, so the additional-seed replication gate is not clean.")
    return "\n".join(lines) + "\n"


def _build_matched_control_report(rows: list[dict[str, Any]], episodes: int) -> str:
    grouped = _group(rows)
    labels = [
        ("flat_dense", "flat_dense"),
        ("recovered_token_dense", "recovered token_dense"),
        ("kl_lss_token_dense", "KL learner-state token_dense"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lines = [
        "# Matched Teacher-Guided Control Report",
        "",
        f"- external evaluation episodes per mode: `{episodes}`",
        "",
        "| Seed | flat_dense | recovered token_dense | KL learner-state token_dense | baseline PPO SARE | KL learner-state SARE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in sorted({key[0] for key in grouped}):
        values = [str(seed)]
        for label, _display in labels:
            values.append(_format_float(_greedy_row(grouped[(seed, label)]).get("eval_success_rate")))
        lines.append("| " + " | ".join(values) + " |")
    mean_recovered = _mean_greedy(grouped, "recovered_token_dense")
    mean_lss_token = _mean_greedy(grouped, "kl_lss_token_dense")
    mean_baseline = _mean_greedy(grouped, "baseline_sare")
    mean_lss_sare = _mean_greedy(grouped, "kl_lss_sare")
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
    if mean_lss_token > mean_recovered and mean_lss_sare > mean_baseline:
        lines.append("- Teacher-guided KL learner-state supervision helps both tokenized and routed students under matched settings.")
    elif mean_lss_sare > mean_baseline and mean_lss_token <= mean_recovered:
        lines.append("- Under matched settings, the main teacher-guided gain appears more specific to SARE than to token_dense.")
    else:
        lines.append("- Under matched settings, the benefit is not specific to routing strongly enough to claim a routed extraction advantage.")
    if mean_lss_sare > mean_lss_token:
        lines.append("- KL learner-state SARE still outperforms the matched teacher-guided token_dense control on mean greedy success.")
    else:
        lines.append("- Matched teacher-guided token_dense is at least as strong as teacher-guided SARE, so the reopened routed claim stays method-first rather than routing-first.")
    return "\n".join(lines) + "\n"


def _build_transfer_report(rows: list[dict[str, Any]], episodes: int) -> str:
    grouped = _group(rows)
    labels = [
        ("flat_dense", "flat_dense"),
        ("recovered_token_dense", "recovered token_dense"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lines = [
        "# KeyCorridor Transfer Report",
        "",
        f"- external evaluation episodes per mode: `{episodes}`",
        "",
        "| Seed | flat_dense | recovered token_dense | baseline PPO SARE | KL learner-state SARE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for seed in sorted({key[0] for key in grouped}):
        values = [str(seed)]
        for label, _display in labels:
            values.append(_format_float(_greedy_row(grouped[(seed, label)]).get("eval_success_rate")))
        lines.append("| " + " | ".join(values) + " |")
    mean_token = _mean_greedy(grouped, "recovered_token_dense")
    mean_sare = _mean_greedy(grouped, "kl_lss_sare")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
        ]
    )
    if mean_sare > 0.0 and mean_sare >= mean_token:
        lines.append("- Weak positive transfer: KL learner-state SARE shows nonzero KeyCorridor greedy success and remains competitive with recovered token_dense under the bounded check.")
    elif mean_sare > 0.0:
        lines.append("- Inconclusive under bounded budget: KL learner-state SARE shows some nonzero KeyCorridor signal but does not stay competitive with recovered token_dense.")
    else:
        lines.append("- No transfer: KL learner-state SARE stays greedy-flat on KeyCorridor under the bounded check.")
    return "\n".join(lines) + "\n"


def reproduction_note(args: argparse.Namespace) -> None:
    baseline_targets = _filter_baselines(Path(args.baseline_root), {"flat_dense", "recovered_token_dense", "baseline_sare"})
    improved_targets = _variant_targets(Path(args.improved_root), {"sare": "kl_lss_sare"})
    rows = _evaluate_targets(baseline_targets + improved_targets, args.device, args.episodes)
    _write_main_process_report(
        rows,
        args.output,
        args.csv,
        _build_reproduction_note(rows, Path(args.baseline_root), Path(args.improved_root), args.episodes),
    )


def additional_seed_report(args: argparse.Namespace) -> None:
    baseline_targets = _filter_baselines(Path(args.baseline_root), {"flat_dense", "recovered_token_dense", "baseline_sare"})
    improved_targets = _variant_targets(Path(args.improved_root), {"sare": "kl_lss_sare"})
    rows = _evaluate_targets(baseline_targets + improved_targets, args.device, args.episodes)
    _write_main_process_report(rows, args.output, args.csv, _build_additional_seed_report(rows, args.episodes))


def matched_control_report(args: argparse.Namespace) -> None:
    baseline_targets = _filter_baselines(Path(args.baseline_root), {"flat_dense", "recovered_token_dense", "baseline_sare"})
    sare_targets = _variant_targets(Path(args.sare_root), {"sare": "kl_lss_sare"})
    token_targets = _variant_targets(Path(args.token_root), {"token_dense": "kl_lss_token_dense"})
    rows = _evaluate_targets(baseline_targets + sare_targets + token_targets, args.device, args.episodes)
    _write_main_process_report(rows, args.output, args.csv, _build_matched_control_report(rows, args.episodes))


def transfer_report(args: argparse.Namespace) -> None:
    baseline_targets = _filter_baselines(Path(args.baseline_root), {"flat_dense", "recovered_token_dense", "baseline_sare"})
    improved_targets = _variant_targets(Path(args.improved_root), {"sare": "kl_lss_sare"})
    rows = _evaluate_targets(baseline_targets + improved_targets, args.device, args.episodes)
    _write_main_process_report(rows, args.output, args.csv, _build_transfer_report(rows, args.episodes))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze claim-hardening learner-state DoorKey experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reproduction = subparsers.add_parser("reproduction-note")
    reproduction.add_argument("--baseline-root", required=True)
    reproduction.add_argument("--improved-root", required=True)
    reproduction.add_argument("--episodes", type=int, default=64)
    reproduction.add_argument("--device", default="auto")
    reproduction.add_argument("--output", required=True)
    reproduction.add_argument("--csv", default=None)

    additional = subparsers.add_parser("additional-seed-report")
    additional.add_argument("--baseline-root", required=True)
    additional.add_argument("--improved-root", required=True)
    additional.add_argument("--episodes", type=int, default=64)
    additional.add_argument("--device", default="auto")
    additional.add_argument("--output", required=True)
    additional.add_argument("--csv", default=None)

    matched = subparsers.add_parser("matched-control-report")
    matched.add_argument("--baseline-root", required=True)
    matched.add_argument("--sare-root", required=True)
    matched.add_argument("--token-root", required=True)
    matched.add_argument("--episodes", type=int, default=64)
    matched.add_argument("--device", default="auto")
    matched.add_argument("--output", required=True)
    matched.add_argument("--csv", default=None)

    transfer = subparsers.add_parser("transfer-report")
    transfer.add_argument("--baseline-root", required=True)
    transfer.add_argument("--improved-root", required=True)
    transfer.add_argument("--episodes", type=int, default=64)
    transfer.add_argument("--device", default="auto")
    transfer.add_argument("--output", required=True)
    transfer.add_argument("--csv", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "reproduction-note":
        reproduction_note(args)
        return
    if args.command == "additional-seed-report":
        additional_seed_report(args)
        return
    if args.command == "transfer-report":
        transfer_report(args)
        return
    matched_control_report(args)


if __name__ == "__main__":
    main()
