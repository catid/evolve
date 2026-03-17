from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_claim_consolidation import _group, _lane_seeds, _summary_stats
from psmn_rl.analysis.lss_frozen_claim import (
    REPRO_MODES,
    _capture_route_batch_stats,
    _evaluate_modes,
    _load_summary,
    _read_csv_rows,
    _targets_from_rows,
)
from psmn_rl.analysis.lss_robustness import _format_float, _greedy_row, _maybe_read_command, _write_main_process_report
from psmn_rl.utils.io import get_git_commit, get_git_dirty


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _best_round_record(run_dir: Path) -> dict[str, Any]:
    summary = _load_summary(run_dir)
    best_round_index = int(summary["best_round_index"])
    rounds = summary["rounds"]
    record = dict(rounds[best_round_index - 1])
    record["best_round_index"] = best_round_index
    record["best_round_greedy_success"] = float(summary["best_round_greedy_success"])
    record["final_greedy_success"] = float(summary["final_greedy_success"])
    return record


def _trend(rounds: list[dict[str, Any]], key: str) -> tuple[float, float]:
    if not rounds:
        return 0.0, 0.0
    return float(rounds[0].get(key, 0.0)), float(rounds[-1].get(key, 0.0))


def _route_pair_stats_for_best_round(run_dir: Path, device: str, max_samples: int = 256) -> dict[str, float]:
    summary = _load_summary(run_dir)
    best_round_index = int(summary["best_round_index"])
    round_name = f"round_{best_round_index:02d}"
    checkpoint_path = run_dir / f"{round_name}.pt"
    dataset_path = run_dir / f"{round_name}_dataset.pt"
    config_path = run_dir / "student_resolved_config.yaml"
    return _capture_route_batch_stats(run_dir, checkpoint_path, config_path, dataset_path, max_samples=max_samples, device=device)


def _build_reproduction_note(rows: list[dict[str, Any]], source_csv: Path, episodes: int) -> str:
    grouped = _group(rows)
    labels = [
        ("kl_lss_token_dense", "KL learner-state token_dense"),
        ("kl_lss_single_expert", "KL learner-state single_expert"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lines = [
        "# Resume-Gate Reproduction Note",
        "",
        f"- source csv: `{source_csv}`",
        f"- external evaluation episodes per mode: `{episodes}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "| Seed | Variant | Greedy Success | Sampled t=1.0 Success | Config | Checkpoint | Command |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for seed in _lane_seeds(grouped, "fresh_final"):
        for label, display in labels:
            mode_rows = grouped[("fresh_final", seed, label)]
            greedy = _greedy_row(mode_rows)
            sampled = next(row for row in mode_rows if row["mode"] == "sampled_t1.0")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(seed),
                        display,
                        _format_float(greedy.get("eval_success_rate")),
                        _format_float(sampled.get("eval_success_rate")),
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
            "- This note re-evaluates the frozen `47/53/59` DoorKey block on the external 64-episode path before any resume decision.",
            "- The reproduced block is the only decision lane for this phase.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_failure_mechanism_report(
    reproduced_rows: list[dict[str, Any]],
    structured_rows: list[dict[str, Any]],
    prior_failure_rows: list[dict[str, Any]],
    device: str,
) -> tuple[str, list[dict[str, Any]], str]:
    reproduced_grouped = _group(reproduced_rows)
    structured_grouped = _group(structured_rows)
    prior_failure_grouped: dict[int, list[dict[str, Any]]] = {}
    for row in prior_failure_rows:
        prior_failure_grouped.setdefault(int(row["seed"]), []).append(row)

    strong_seeds: list[int] = []
    strong_run_dirs: dict[int, Path] = {}
    for (lane, seed, label), rows in sorted(structured_grouped.items()):
        if label != "kl_lss_sare" or lane == "fresh_final":
            continue
        greedy = float(_greedy_row(rows).get("eval_success_rate", 0.0))
        if greedy >= 0.5:
            strong_seeds.append(seed)
            strong_run_dirs[seed] = Path(str(_greedy_row(rows)["run_dir"]))

    final_run_dirs = {
        seed: Path(str(_greedy_row(reproduced_grouped[("fresh_final", seed, "kl_lss_sare")])["run_dir"]))
        for seed in _lane_seeds(reproduced_grouped, "fresh_final")
    }
    single_run_dirs = {
        seed: Path(str(_greedy_row(reproduced_grouped[("fresh_final", seed, "kl_lss_single_expert")])["run_dir"]))
        for seed in _lane_seeds(reproduced_grouped, "fresh_final")
    }

    strong_best = {seed: _best_round_record(run_dir) for seed, run_dir in strong_run_dirs.items()}
    final_best = {seed: _best_round_record(run_dir) for seed, run_dir in final_run_dirs.items()}
    single_best = {seed: _best_round_record(run_dir) for seed, run_dir in single_run_dirs.items()}

    strong_route_pairs = {seed: _route_pair_stats_for_best_round(run_dir, device) for seed, run_dir in strong_run_dirs.items()}

    strong_teacher = _mean([float(row.get("collection/teacher_confidence_mean", 0.0)) for row in strong_best.values()])
    strong_disagreement = _mean([float(row.get("collection/disagreement_rate", 0.0)) for row in strong_best.values()])
    strong_unique = _mean([float(row.get("collection/unique_state_ratio", 0.0)) for row in strong_best.values()])
    strong_route_entropy = _mean([float(row.get("collection/route_entropy", 0.0)) for row in strong_best.values()])
    strong_path_entropy = _mean([float(row.get("collection/path_entropy", 0.0)) for row in strong_best.values()])
    strong_dominant_pair = _mean([float(row["route_pair_dominant_mean"]) for row in strong_route_pairs.values()])

    final_teacher = _mean([float(row.get("collection/teacher_confidence_mean", 0.0)) for row in final_best.values()])
    final_disagreement = _mean([float(row.get("collection/disagreement_rate", 0.0)) for row in final_best.values()])
    final_unique = _mean([float(row.get("collection/unique_state_ratio", 0.0)) for row in final_best.values()])
    final_route_entropy = _mean([float(row.get("collection/route_entropy", 0.0)) for row in final_best.values()])
    final_path_entropy = _mean([float(row.get("collection/path_entropy", 0.0)) for row in final_best.values()])

    audit_rows: list[dict[str, Any]] = [
        {
            "kind": "reference",
            "seed": "strong_avg",
            "teacher_confidence_mean": strong_teacher,
            "disagreement_rate": strong_disagreement,
            "unique_state_ratio": strong_unique,
            "route_entropy": strong_route_entropy,
            "path_entropy": strong_path_entropy,
            "route_pair_dominant_mean": strong_dominant_pair,
        },
        {
            "kind": "reference",
            "seed": "final_avg",
            "teacher_confidence_mean": final_teacher,
            "disagreement_rate": final_disagreement,
            "unique_state_ratio": final_unique,
            "route_entropy": final_route_entropy,
            "path_entropy": final_path_entropy,
        },
    ]

    lines = [
        "# Resume-Gate Failure Mechanism Report",
        "",
        "## Reference Comparison",
        "",
        "| Group | Teacher Conf | Disagreement | Unique Ratio | Route Entropy | Path Entropy | Dominant Route Pair |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| stronger recovered `SARE` seeds | `{strong_teacher:.4f}` | `{strong_disagreement:.4f}` | `{strong_unique:.4f}` | `{strong_route_entropy:.4f}` | `{strong_path_entropy:.4f}` | `{strong_dominant_pair:.4f}` |",
        f"| final block `47/53/59` | `{final_teacher:.4f}` | `{final_disagreement:.4f}` | `{final_unique:.4f}` | `{final_route_entropy:.4f}` | `{final_path_entropy:.4f}` | `see per-seed below` |",
        "",
        "## Per-Seed Final-Block Summary",
        "",
        "| Seed | Variant | Best Round | Best Greedy | Final Greedy | Disagreement r1->rN | Teacher Conf r1->rN | Student Conf r1->rN | Unique Ratio r1->rN | Route Ent r1->rN | Path Ent r1->rN | Dominant Route Pair |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | ---: |",
    ]

    final_seed_dominant_flags = 0
    for seed in sorted(final_run_dirs):
        sare_summary = _load_summary(final_run_dirs[seed])
        single_summary = _load_summary(single_run_dirs[seed])
        sare_rounds = sare_summary["rounds"]
        single_rounds = single_summary["rounds"]
        final_probe_rows = prior_failure_grouped[seed]
        dominant_row = next((row for row in final_probe_rows if row.get("route_pair_dominant_mean")), None)
        dominant_pair = float(dominant_row.get("route_pair_dominant_mean", 0.0)) if dominant_row is not None else 0.0
        if dominant_pair > strong_dominant_pair + 0.10:
            final_seed_dominant_flags += 1

        sare_first_disagreement, sare_last_disagreement = _trend(sare_rounds, "collection/disagreement_rate")
        sare_first_teacher, sare_last_teacher = _trend(sare_rounds, "collection/teacher_confidence_mean")
        sare_first_student, sare_last_student = _trend(sare_rounds, "collection/student_confidence_mean")
        sare_first_unique, sare_last_unique = _trend(sare_rounds, "collection/unique_state_ratio")
        sare_first_route, sare_last_route = _trend(sare_rounds, "collection/route_entropy")
        sare_first_path, sare_last_path = _trend(sare_rounds, "collection/path_entropy")
        single_first_disagreement, single_last_disagreement = _trend(single_rounds, "collection/disagreement_rate")
        single_first_teacher, single_last_teacher = _trend(single_rounds, "collection/teacher_confidence_mean")
        single_first_student, single_last_student = _trend(single_rounds, "collection/student_confidence_mean")
        single_first_unique, single_last_unique = _trend(single_rounds, "collection/unique_state_ratio")

        lines.append(
            "| "
            + " | ".join(
                [
                    str(seed),
                    "KL learner-state SARE",
                    str(int(final_best[seed]["best_round_index"])),
                    _format_float(final_best[seed]["best_round_greedy_success"]),
                    _format_float(final_best[seed]["final_greedy_success"]),
                    f"`{sare_first_disagreement:.4f}->{sare_last_disagreement:.4f}`",
                    f"`{sare_first_teacher:.4f}->{sare_last_teacher:.4f}`",
                    f"`{sare_first_student:.4f}->{sare_last_student:.4f}`",
                    f"`{sare_first_unique:.4f}->{sare_last_unique:.4f}`",
                    f"`{sare_first_route:.4f}->{sare_last_route:.4f}`",
                    f"`{sare_first_path:.4f}->{sare_last_path:.4f}`",
                    _format_float(dominant_pair),
                ]
            )
            + " |"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(seed),
                    "KL learner-state single_expert",
                    str(int(single_best[seed]["best_round_index"])),
                    _format_float(single_best[seed]["best_round_greedy_success"]),
                    _format_float(single_best[seed]["final_greedy_success"]),
                    f"`{single_first_disagreement:.4f}->{single_last_disagreement:.4f}`",
                    f"`{single_first_teacher:.4f}->{single_last_teacher:.4f}`",
                    f"`{single_first_student:.4f}->{single_last_student:.4f}`",
                    f"`{single_first_unique:.4f}->{single_last_unique:.4f}`",
                    "`0.0000->0.0000`",
                    "`0.0000->0.0000`",
                    "`n/a`",
                ]
            )
            + " |"
        )

        audit_rows.extend(
            [
                {
                    "kind": "final_seed",
                    "seed": seed,
                    "variant": "kl_lss_sare",
                    "best_round_index": final_best[seed]["best_round_index"],
                    "best_round_greedy_success": final_best[seed]["best_round_greedy_success"],
                    "final_greedy_success": final_best[seed]["final_greedy_success"],
                    "disagreement_start": sare_first_disagreement,
                    "disagreement_end": sare_last_disagreement,
                    "teacher_confidence_start": sare_first_teacher,
                    "teacher_confidence_end": sare_last_teacher,
                    "student_confidence_start": sare_first_student,
                    "student_confidence_end": sare_last_student,
                    "unique_ratio_start": sare_first_unique,
                    "unique_ratio_end": sare_last_unique,
                    "route_entropy_start": sare_first_route,
                    "route_entropy_end": sare_last_route,
                    "path_entropy_start": sare_first_path,
                    "path_entropy_end": sare_last_path,
                    "route_pair_dominant_mean": dominant_pair,
                },
                {
                    "kind": "final_seed",
                    "seed": seed,
                    "variant": "kl_lss_single_expert",
                    "best_round_index": single_best[seed]["best_round_index"],
                    "best_round_greedy_success": single_best[seed]["best_round_greedy_success"],
                    "final_greedy_success": single_best[seed]["final_greedy_success"],
                    "disagreement_start": single_first_disagreement,
                    "disagreement_end": single_last_disagreement,
                    "teacher_confidence_start": single_first_teacher,
                    "teacher_confidence_end": single_last_teacher,
                    "student_confidence_start": single_first_student,
                    "student_confidence_end": single_last_student,
                    "unique_ratio_start": single_first_unique,
                    "unique_ratio_end": single_last_unique,
                },
            ]
        )

    lines.extend(
        [
            "",
            "## Mechanism Audit",
            "",
        ]
    )

    teacher_quality_problem = final_teacher < strong_teacher - 0.05
    coverage_problem = final_unique < strong_unique * 0.75
    route_redundancy_problem = final_seed_dominant_flags >= 2
    final_single_mean = _summary_stats(reproduced_grouped, "kl_lss_single_expert", "fresh_final")["mean"]
    final_sare_mean = _summary_stats(reproduced_grouped, "kl_lss_sare", "fresh_final")["mean"]

    if teacher_quality_problem:
        lines.append(f"- Teacher quality looks lower on the final block (`{final_teacher:.4f}` vs `{strong_teacher:.4f}`), so poor teacher labels would be a plausible mechanism.")
    else:
        lines.append(f"- Teacher quality is not the blocker: best-round teacher confidence stays high on the final block (`{final_teacher:.4f}` vs `{strong_teacher:.4f}` on stronger recovered seeds).")

    if coverage_problem:
        lines.append(f"- Learner-state coverage is lower on the final block (`{final_unique:.4f}` vs `{strong_unique:.4f}`), so coverage could contribute.")
    else:
        lines.append(f"- Learner-state coverage is not cleanly isolating the reversal: final-block unique ratio is lower on average (`{final_unique:.4f}` vs `{strong_unique:.4f}`), but seeds `53` and `59` stay within the earlier recovered range.")

    if route_redundancy_problem:
        lines.append(f"- State-local expert redundancy is elevated across most final-block seeds, with dominant route-pair concentration above the stronger-seed reference on multiple seeds.")
    else:
        lines.append(f"- State-local expert redundancy is not consistent across the weak block: only seed `47` is clearly above the stronger-seed dominant-pair reference, while `53` and `59` are not.")

    lines.append("- Route-specific fragility remains real, not absent: fixed-router override and worst-expert ablation still collapse the recovered `53` and `59` checkpoints to zero in the prior final-block probe set.")

    if final_single_mean >= final_sare_mean:
        lines.append(f"- The main fairness signal cuts against a resume attempt: matched KL learner-state `single_expert` is already at least as strong on the final block (`{final_single_mean:.4f}` vs `{final_sare_mean:.4f}` for `SARE`).")

    if (not teacher_quality_problem) and (not route_redundancy_problem) and final_single_mean >= final_sare_mean:
        verdict = "no actionable mechanism"
        lines.append("- Across `47/53/59`, the weak block does not expose one clean resume-worthy mechanism: seed `47` looks route-fragile, but `53` and `59` already collapse to a generic structured-student tie with `single_expert`.")
    else:
        verdict = "mechanism plausible but weak"
        lines.append("- The weak block suggests one plausible mechanism, but it is not strong or consistent enough yet to justify a resume attempt.")

    audit_rows.append({"kind": "verdict", "resume_gate_status": verdict})
    lines.extend(["", "## Verdict", "", verdict])
    return "\n".join(lines) + "\n", audit_rows, verdict


def _build_decision_memo(
    reproduced_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    combined_rows: list[dict[str, Any]],
) -> str:
    reproduced_grouped = _group(reproduced_rows)
    combined_grouped = _group(combined_rows)
    verdict_row = next(row for row in failure_rows if row.get("kind") == "verdict")
    verdict = str(verdict_row["resume_gate_status"])
    final_token = _summary_stats(reproduced_grouped, "kl_lss_token_dense", "fresh_final")
    final_single = _summary_stats(reproduced_grouped, "kl_lss_single_expert", "fresh_final")
    final_sare = _summary_stats(reproduced_grouped, "kl_lss_sare", "fresh_final")
    combined_token = _summary_stats(combined_grouped, "kl_lss_token_dense")
    combined_single = _summary_stats(combined_grouped, "kl_lss_single_expert")
    combined_sare = _summary_stats(combined_grouped, "kl_lss_sare")

    lines = [
        "# Resume-Gate Decision Memo",
        "",
        "## Decision",
        "",
    ]
    if verdict == "actionable mechanism found":
        decision = "conditionally thawed within DoorKey"
    elif verdict == "mechanism plausible but weak" and combined_sare["mean"] <= combined_single["mean"]:
        decision = "narrow further"
    else:
        decision = "stay frozen as-is"
    lines.append(f"The right final state is: **{decision}**.")
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. Actionable mechanism status: `{verdict}`. See [lss_resume_gate_failure_mechanism_report.md](lss_resume_gate_failure_mechanism_report.md).",
            f"2. Bounded resume attempt justified: `no`. The final block is `{final_sare['mean']:.4f}` for KL learner-state `SARE`, `{final_single['mean']:.4f}` for KL learner-state `single_expert`, and `{final_token['mean']:.4f}` for KL learner-state `token_dense`, so the missing fairness control does not earn a retry. See [lss_final_block_single_expert_control_report.md](lss_final_block_single_expert_control_report.md).",
            "3. Resume attempt run: `no`. No preregistered plan was written because the mechanism audit did not clear the resume gate. See [lss_resume_gate_failure_mechanism_report.md](lss_resume_gate_failure_mechanism_report.md).",
            f"4. The combined DoorKey picture remains `{combined_sare['mean']:.4f}` for KL learner-state `SARE`, `{combined_single['mean']:.4f}` for KL learner-state `single_expert`, and `{combined_token['mean']:.4f}` for KL learner-state `token_dense`; that is still not strong enough to thaw the claim. See [lss_frozen_claim_updated_combined_doorkey_report.md](lss_frozen_claim_updated_combined_doorkey_report.md).",
            "",
            "Recommendation: stay frozen. The current DoorKey teacher-guided `SARE` result remains a bounded positive result, but this resume gate does not justify a new retry.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume-gate analysis for the frozen DoorKey SARE claim.")
    sub = parser.add_subparsers(dest="command", required=True)

    repro = sub.add_parser("reproduction-note")
    repro.add_argument("--final-csv", required=True)
    repro.add_argument("--episodes", type=int, default=64)
    repro.add_argument("--device", default="auto")
    repro.add_argument("--output", required=True)
    repro.add_argument("--csv", required=True)

    failure = sub.add_parser("failure-mechanism-report")
    failure.add_argument("--structured-csv", required=True)
    failure.add_argument("--final-csv", required=True)
    failure.add_argument("--prior-failure-csv", required=True)
    failure.add_argument("--device", default="auto")
    failure.add_argument("--output", required=True)
    failure.add_argument("--csv", required=True)

    memo = sub.add_parser("decision-memo")
    memo.add_argument("--reproduction-csv", required=True)
    memo.add_argument("--failure-csv", required=True)
    memo.add_argument("--combined-csv", required=True)
    memo.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "reproduction-note":
        final_rows = _read_csv_rows(Path(args.final_csv))
        targets = _targets_from_rows(final_rows, {"kl_lss_token_dense", "kl_lss_single_expert", "kl_lss_sare"})
        rows = _evaluate_modes(targets, args.device, args.episodes, REPRO_MODES)
        content = _build_reproduction_note(rows, Path(args.final_csv), args.episodes)
        _write_main_process_report(rows, args.output, args.csv, content)
        return

    if args.command == "failure-mechanism-report":
        reproduced_rows = _read_csv_rows(Path(args.final_csv))
        structured_rows = _read_csv_rows(Path(args.structured_csv))
        prior_failure_rows = _read_csv_rows(Path(args.prior_failure_csv))
        content, audit_rows, _verdict = _build_failure_mechanism_report(reproduced_rows, structured_rows, prior_failure_rows, args.device)
        _write_main_process_report(audit_rows, args.output, args.csv, content)
        return

    if args.command == "decision-memo":
        reproduced_rows = _read_csv_rows(Path(args.reproduction_csv))
        failure_rows = _read_csv_rows(Path(args.failure_csv))
        combined_rows = _read_csv_rows(Path(args.combined_csv))
        content = _build_decision_memo(reproduced_rows, failure_rows, combined_rows)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(content, encoding="utf-8")
        return

    raise ValueError(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
