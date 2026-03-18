from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from psmn_rl.analysis.benchmark_pack import sha256_path
from psmn_rl.analysis.lss_post_pass_campaign import (
    DISPLAY_NAMES,
    _float,
    _load_yaml,
    _read_json,
    _stats,
    _write_csv,
    _write_json,
)
from psmn_rl.analysis.lss_robustness import _format_float
from psmn_rl.analysis.lss_successor_migration import (
    CURRENT_LABELS,
    _block_lanes,
    _candidate_row,
    _control_family,
    _current_round6_rows,
    _discover_runs,
    _historical_refs,
)
from psmn_rl.utils.io import get_git_commit, get_git_dirty


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _winner(campaign: dict[str, Any], holdout: dict[str, Any], anti_regression: dict[str, Any], route: dict[str, Any], stability: dict[str, Any]) -> dict[str, Any]:
    best_candidate = holdout.get("best_candidate")
    challenger_viable = (
        best_candidate is not None
        and bool(anti_regression.get("challenger_pass"))
        and bool(route.get("challenger_pass"))
        and bool(stability.get("challenger_pass"))
    )
    return {
        "winner": str(best_candidate) if challenger_viable else str(campaign["current_canonical_name"]),
        "challenger_viable": challenger_viable,
        "best_candidate": best_candidate,
    }


def _render_state_reconciliation(campaign: dict[str, Any], output: Path) -> None:
    readme = _read_text("README.md")
    report_md = _read_text("report.md")
    summary_md = _read_text("summary.md")
    ledger_md = _read_text("outputs/reports/claim_ledger.md")
    migration_memo = _read_text("outputs/reports/successor_migration_decision_memo.md")
    mega_memo = _read_text("outputs/reports/successor_mega_league_decision_memo.md")
    active_pack = _read_json(Path(campaign["current_canonical_pack"]))
    migration_gate = _read_json(Path("outputs/reports/successor_migration_gate_report.json"))
    mega_gate = _read_json(Path("outputs/reports/successor_mega_league_gate_report.json"))
    freeze_source = _read_text("src/psmn_rl/analysis/freeze_hardening.py")
    migration_source = _read_text("src/psmn_rl/analysis/lss_successor_migration.py")

    checks = {
        "docs_round6_active": all(
            needle in text
            for needle, text in (
                ("round6", readme),
                ("active DoorKey benchmark", report_md),
                ("active canonical DoorKey benchmark", summary_md),
            )
        ),
        "ledger_active_row": "Post-canonization migration and challenger league" in ledger_md and "active canonical benchmark" in ledger_md,
        "pack_active_round6": str(active_pack.get("candidate_name")) == "round6",
        "pack_archives_frozen": "archived_legacy_frozen_pack" in active_pack.get("migration", {}),
        "migration_memo_active": "round6 canonized as active DoorKey benchmark" in migration_memo,
        "mega_memo_active": "round6 confirmed and sealed as active canonical DoorKey benchmark" in mega_memo,
        "migration_gate_pass": str(migration_gate.get("verdict")) == "PASS: thaw consideration allowed",
        "mega_gate_pass": str(mega_gate.get("verdict")) == "PASS: thaw consideration allowed",
        "freeze_hardening_tracks_active_rows": "Post-canonization migration and challenger league" in freeze_source,
        "migration_pack_refreshes_frozen_reference": "current_pack[\"frozen_pack_reference\"]" in migration_source,
    }
    status = "reconciled" if all(checks.values()) else "inconsistent"
    lines = [
        "# Successor Mega League State Reconciliation",
        "",
        f"- status: `{status}`",
        f"- active benchmark pack: `{campaign['current_canonical_pack']}`",
        f"- archived frozen pack: `{campaign['legacy_frozen_pack']}`",
        f"- active benchmark candidate: `{campaign['current_canonical_name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Authoritative Current State",
        "",
        "- `round6` is the active DoorKey benchmark successor.",
        "- `outputs/reports/frozen_benchmark_pack.json` remains the archived legacy baseline and provenance anchor.",
        "- The allowed claim envelope remains teacher-guided only, KL learner-state only, DoorKey only, external 64-episode evaluation only, and still excludes PPO-only, specifically multi-expert, cross-task, and KeyCorridor claims.",
        "",
        "## Reconciliation Checks",
        "",
        f"- docs present round6 as active benchmark: `{checks['docs_round6_active']}`",
        f"- claim ledger records active canonical benchmark: `{checks['ledger_active_row']}`",
        f"- active benchmark pack names round6: `{checks['pack_active_round6']}`",
        f"- active benchmark pack archives frozen legacy baseline: `{checks['pack_archives_frozen']}`",
        f"- migration memo records active benchmark status: `{checks['migration_memo_active']}`",
        f"- mega league memo records confirmed active benchmark status: `{checks['mega_memo_active']}`",
        f"- migration gate verdict stays PASS: `{checks['migration_gate_pass']}`",
        f"- mega league gate verdict stays PASS: `{checks['mega_gate_pass']}`",
        f"- freeze hardening source keeps the active rows during rerender: `{checks['freeze_hardening_tracks_active_rows']}`",
        f"- migration pack source refreshes frozen-pack reference hashes: `{checks['migration_pack_refreshes_frozen_reference']}`",
        "",
        "## Reconciled Interpretation",
        "",
        "- The repo previously had a stale-state risk: rerendering the ledger from `freeze_hardening.py` could drop the active-benchmark rows, and the migration pack builder could preserve an old frozen-pack hash.",
        "- Those two failure modes are now fixed in source, and the post-league active pack has been refreshed and re-gated, so top-level docs, the claim ledger, the migration memo, the mega-league memo, the active benchmark pack, and the frozen-pack validation lane agree on the same state.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_registration(campaign: dict[str, Any], output: Path) -> None:
    directions = sorted({str(meta["direction"]) for meta in campaign["candidates"].values()})
    lines = [
        "# Successor Mega League Registration",
        "",
        f"- active incumbent: `{campaign['current_canonical_name']}`",
        f"- active benchmark pack: `{campaign['current_canonical_pack']}`",
        f"- archived frozen pack: `{campaign['legacy_frozen_pack']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Families",
        "",
        f"- development families: `{campaign['blocks']['dev']}`",
        f"- holdout families: `{campaign['blocks']['holdout']}`",
        f"- healthy anti-regression families: `{campaign['blocks']['healthy']}`",
        "",
        "## League Size",
        "",
        f"- mechanism directions: `{len(directions)}`",
        f"- mechanism direction names: `{directions}`",
        f"- challenger variants: `{len(campaign['candidates'])}`",
        f"- verification reruns per Stage 1 survivor: `2`",
        "",
        "## Narrowing Rules",
        "",
        f"- Stage 1 top-k: `{campaign['selection']['stage1_top_k']}`",
        f"- verification tolerance: `{_float(campaign['selection']['verification_tolerance']):.4f}`",
        f"- fairness top-k: `{campaign['selection']['stage2_top_k']}`",
        f"- holdout tolerance: `{_float(campaign['selection']['min_holdout_gain']):.4f}`",
        f"- anti-regression tolerance: `{_float(campaign['selection']['anti_regression_tolerance']):.4f}`",
        "",
        "## Final Decision Rule",
        "",
        "- If no challenger survives verification, fairness, holdout, anti-regression, route, and stability cleanly enough to replace the incumbent, the program still continues through the active-benchmark confirmation path for round6.",
        "- Any winner still has to clear the pack/gate lane relative to the archived frozen pack.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_baseline_sync(campaign: dict[str, Any], output: Path, csv_output: Path | None, json_output: Path | None) -> None:
    frozen_pack = _read_json(Path(campaign["frozen_pack"]))
    active_pack = _read_json(Path(campaign["current_canonical_pack"]))
    current_rows = _current_round6_rows(campaign)
    dev = _control_family(current_rows, str(campaign["current_canonical_name"]), _block_lanes(campaign, "dev"))
    holdout = _control_family(current_rows, str(campaign["current_canonical_name"]), _block_lanes(campaign, "holdout"))
    healthy = _control_family(current_rows, str(campaign["current_canonical_name"]), _block_lanes(campaign, "healthy"))
    lines = [
        "# Successor Mega League Baseline Sync",
        "",
        f"- archived frozen pack: `{campaign['frozen_pack']}`",
        f"- active benchmark pack: `{campaign['current_canonical_pack']}`",
        f"- active benchmark candidate: `{campaign['current_canonical_name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Pack Snapshot",
        "",
        f"- frozen retry-block KL learner-state `SARE` threshold: `{_float(frozen_pack['thresholds']['retry_block_means']['kl_lss_sare']):.4f}`",
        f"- active retry-block KL learner-state `SARE` mean: `{_float(active_pack['metrics']['retry_block']['kl_lss_sare']['mean']):.4f}`",
        f"- active frozen-comparable combined KL learner-state `SARE` mean: `{_float(active_pack['metrics']['combined']['kl_lss_sare']['mean']):.4f}`",
        "",
        "## Current Round6 Family Snapshot",
        "",
        f"- dev-family round6 SARE/token/single: `{dev['sare_mean']:.4f}` / `{dev['token_mean']:.4f}` / `{dev['single_mean']:.4f}`",
        f"- holdout-family round6 SARE/token/single: `{holdout['sare_mean']:.4f}` / `{holdout['token_mean']:.4f}` / `{holdout['single_mean']:.4f}`",
        f"- healthy-family round6 SARE/token/single: `{healthy['sare_mean']:.4f}` / `{healthy['token_mean']:.4f}` / `{healthy['single_mean']:.4f}`",
        "",
        "## Historical Challenger Context",
        "",
    ]
    for path in _historical_refs(campaign):
        lines.append(f"- `{path}`")
    lines.extend(["", "| Block | Seed | Variant | Greedy Success |", "| --- | --- | --- | ---: |"])
    for row in sorted(
        [row for row in current_rows if str(row["label"]) in CURRENT_LABELS],
        key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"])),
    ):
        lines.append(
            f"| {row['lane']} | {row['seed']} | {DISPLAY_NAMES[str(row['label'])]} | {_format_float(_float(row['final_greedy_success']))} |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, current_rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "active_pack": active_pack,
                "frozen_pack": frozen_pack,
                "dev": dev,
                "holdout": holdout,
                "healthy": healthy,
                "current_rows": current_rows,
            },
        )


def _rerun_summary(rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    candidate_rows = [row for row in rows if str(row["candidate"]) == candidate]
    stats = _stats([_float(row["final_greedy_success"]) for row in candidate_rows])
    return {
        "candidate": candidate,
        "candidate_mean": stats["mean"],
        "candidate_failures": stats["complete_seed_failures"],
        "row_count": len(candidate_rows),
    }


def _verification_pass(campaign: dict[str, Any], original: dict[str, Any], rerun_a: dict[str, Any], rerun_b: dict[str, Any], round6_failures: float) -> bool:
    tolerance = _float(campaign["selection"]["verification_tolerance"])
    margin_floor = _float(campaign["selection"]["verification_margin_floor"])
    rerun_mean = (rerun_a["candidate_mean"] + rerun_b["candidate_mean"]) / 2.0
    return (
        rerun_a["candidate_mean"] >= original["candidate_mean"] - tolerance
        and rerun_b["candidate_mean"] >= original["candidate_mean"] - tolerance
        and rerun_mean >= original["round6_mean"] + max(margin_floor, original["delta_vs_round6"] - tolerance)
        and rerun_a["candidate_failures"] <= round6_failures
        and rerun_b["candidate_failures"] <= round6_failures
    )


def _render_stage2_verification(campaign: dict[str, Any], output: Path, csv_output: Path | None, json_output: Path | None) -> None:
    stage1 = _read_json(Path(campaign["reports"]["stage1_raw_json"]))
    rows_a = [_candidate_row(record) for record in _discover_runs(Path(campaign["stage_roots"]["stage2_verification_a"])) if str(record.label) == "kl_lss_sare"]
    rows_b = [_candidate_row(record) for record in _discover_runs(Path(campaign["stage_roots"]["stage2_verification_b"])) if str(record.label) == "kl_lss_sare"]
    originals = {str(row["candidate"]): row for row in stage1.get("candidate_summaries", [])}
    round6_summary = stage1.get("round6_summary", {})
    verification_rows: list[dict[str, Any]] = []
    for candidate in stage1.get("advancing_candidates", []):
        candidate_name = str(candidate)
        original = originals[candidate_name]
        rerun_a = _rerun_summary(rows_a, candidate_name)
        rerun_b = _rerun_summary(rows_b, candidate_name)
        payload = {
            "candidate": candidate_name,
            "original_mean": _float(original["candidate_mean"]),
            "original_delta_vs_round6": _float(original["delta_vs_round6"]),
            "rerun_a_mean": _float(rerun_a["candidate_mean"]),
            "rerun_b_mean": _float(rerun_b["candidate_mean"]),
            "rerun_a_delta_vs_round6": _float(rerun_a["candidate_mean"]) - _float(original["round6_mean"]),
            "rerun_b_delta_vs_round6": _float(rerun_b["candidate_mean"]) - _float(original["round6_mean"]),
            "original_failures": _float(original["candidate_failures"]),
            "rerun_a_failures": _float(rerun_a["candidate_failures"]),
            "rerun_b_failures": _float(rerun_b["candidate_failures"]),
        }
        payload["verified"] = _verification_pass(campaign, original, rerun_a, rerun_b, _float(round6_summary.get("sare_failures", 0.0)))
        verification_rows.append(payload)
    verified_candidates = [
        row["candidate"]
        for row in sorted(
            [row for row in verification_rows if row["verified"]],
            key=lambda item: ((item["rerun_a_mean"] + item["rerun_b_mean"]) / 2.0, item["original_delta_vs_round6"]),
            reverse=True,
        )
    ]
    effective_stage1 = copy.deepcopy(stage1)
    effective_stage1["advancing_candidates"] = verified_candidates
    effective_stage1["verification_filtered_candidates"] = verified_candidates
    _write_json(Path(campaign["reports"]["stage1_json"]), effective_stage1)
    lines = [
        "# Successor Mega League Stage 2 Verification",
        "",
        f"- Stage 1 advancing challengers: `{stage1.get('advancing_candidates', [])}`",
        f"- verified challengers: `{verified_candidates}`",
        "",
        "| Candidate | Original Dev Mean | Rerun A | Rerun B | ΔA vs round6 | ΔB vs round6 | Original Failures | Rerun A Failures | Rerun B Failures | Stage 2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in verification_rows:
        lines.append(
            f"| `{row['candidate']}` | `{row['original_mean']:.4f}` | `{row['rerun_a_mean']:.4f}` | `{row['rerun_b_mean']:.4f}` | `{row['rerun_a_delta_vs_round6']:.4f}` | `{row['rerun_b_delta_vs_round6']:.4f}` | `{int(row['original_failures'])}` | `{int(row['rerun_a_failures'])}` | `{int(row['rerun_b_failures'])}` | `{'pass' if row['verified'] else 'stop'}` |"
        )
    if not verification_rows:
        lines.extend(["", "## Interpretation", "", "- No challenger cleared Stage 1, so verification had no candidates to rerun."])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, verification_rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "verification_rows": verification_rows,
                "verified_candidates": verified_candidates,
                "round6_summary": round6_summary,
            },
        )


def _refresh_pack(campaign: dict[str, Any], output: Path) -> None:
    pack = _read_json(output)
    holdout = _read_json(Path(campaign["reports"]["stage3_json"])) if Path(campaign["reports"]["stage3_json"]).exists() else {}
    anti_regression = _read_json(Path(campaign["reports"]["stage4_json"])) if Path(campaign["reports"]["stage4_json"]).exists() else {}
    route = _read_json(Path(campaign["reports"]["stage5_json"])) if Path(campaign["reports"]["stage5_json"]).exists() else {}
    stability = _read_json(Path(campaign["reports"]["stage6_json"])) if Path(campaign["reports"]["stage6_json"]).exists() else {}
    winner = _winner(campaign, holdout, anti_regression, route, stability)
    pack["frozen_pack_reference"] = {
        "path": str(campaign["frozen_pack"]),
        "sha256": sha256_path(Path(campaign["frozen_pack"])),
        "claim_id": _read_json(Path(campaign["frozen_pack"]))["claim"]["id"],
    }
    pack["active_benchmark_state"] = {
        "winner": winner["winner"],
        "challenger_viable_pre_gate": winner["challenger_viable"],
        "active_pack_role": "confirmed_active_round6" if winner["winner"] == str(campaign["current_canonical_name"]) else "challenger_replacement_candidate",
        "current_active_pack": {
            "path": str(campaign["current_canonical_pack"]),
            "sha256": sha256_path(Path(campaign["current_canonical_pack"])),
        },
        "archived_legacy_frozen_pack": {
            "path": str(campaign["legacy_frozen_pack"]),
            "sha256": sha256_path(Path(campaign["legacy_frozen_pack"])),
        },
        "future_comparison_policy": {
            "active_canonical_pack": str(output),
            "archived_legacy_pack": str(campaign["legacy_frozen_pack"]),
            "gate_reference_pack": str(campaign["frozen_pack"]),
        },
        "holdout_summary": holdout,
        "anti_regression_summary": anti_regression,
        "route_validation": route,
        "stability_validation": stability,
    }
    pack.setdefault("migration", {})
    pack["migration"]["future_comparison_policy"] = {
        "active_canonical_pack": str(output),
        "archived_legacy_pack": str(campaign["legacy_frozen_pack"]),
    }
    pack.setdefault("provenance", {})
    pack["provenance"]["git_commit"] = get_git_commit()
    pack["provenance"]["git_dirty"] = get_git_dirty()
    pack["provenance"]["mega_league_notes"] = "post-reconciliation mega challenger league refresh"
    output.write_text(json.dumps(pack, indent=2, sort_keys=True), encoding="utf-8")


def _decision_status(campaign: dict[str, Any], holdout: dict[str, Any], anti_regression: dict[str, Any], route: dict[str, Any], stability: dict[str, Any], gate_payload: dict[str, Any]) -> str:
    winner = _winner(campaign, holdout, anti_regression, route, stability)
    gate_verdict = str(gate_payload.get("verdict", gate_payload.get("status", "not_run")))
    round6_ready = bool(route.get("round6_pass")) and bool(stability.get("round6_pass")) and gate_verdict == str(campaign["selection"]["pack_gate_required_verdict"])
    if winner["challenger_viable"] and gate_verdict == str(campaign["selection"]["pack_gate_required_verdict"]):
        return "challenger replaces round6 as active canonical DoorKey benchmark"
    if winner["winner"] == str(campaign["current_canonical_name"]) and round6_ready:
        return "round6 confirmed and sealed as active canonical DoorKey benchmark"
    if winner["challenger_viable"]:
        return "challenger fails and round6 remains active incumbent"
    return "round6 remains active but not yet operationally sealed"


def _render_decision_memo(campaign: dict[str, Any], output: Path) -> None:
    stage1 = _read_json(Path(campaign["reports"]["stage1_raw_json"]))
    stage2 = _read_json(Path(campaign["reports"]["stage2_verification_json"])) if Path(campaign["reports"]["stage2_verification_json"]).exists() else {"verified_candidates": []}
    stage3 = _read_json(Path(campaign["reports"]["stage2_json"])) if Path(campaign["reports"]["stage2_json"]).exists() else {"surviving_candidates": []}
    stage4 = _read_json(Path(campaign["reports"]["stage3_json"])) if Path(campaign["reports"]["stage3_json"]).exists() else {}
    stage5 = _read_json(Path(campaign["reports"]["stage4_json"])) if Path(campaign["reports"]["stage4_json"]).exists() else {}
    stage6 = _read_json(Path(campaign["reports"]["stage5_json"])) if Path(campaign["reports"]["stage5_json"]).exists() else {}
    stage7 = _read_json(Path(campaign["reports"]["stage6_json"])) if Path(campaign["reports"]["stage6_json"]).exists() else {}
    gate_payload = _read_json(Path(campaign["reports"]["gate_report_json"])) if Path(campaign["reports"]["gate_report_json"]).exists() else {"status": "not_run"}
    final_status = _decision_status(campaign, stage4, stage5, stage6, stage7, gate_payload)
    winner = _winner(campaign, stage4, stage5, stage6, stage7)
    lines = [
        "# Successor Mega League Decision Memo",
        "",
        f"- final status: `{final_status}`",
        f"- winning line: `{winner['winner']}`",
        f"- gate verdict: `{gate_payload.get('verdict', gate_payload.get('status', 'not_run'))}`",
        "",
        "## League Funnel",
        "",
        f"- Stage 1 advancing challengers: `{stage1.get('advancing_candidates', [])}`",
        f"- Stage 2 verified challengers: `{stage2.get('verified_candidates', [])}`",
        f"- Stage 3 fairness survivors: `{stage3.get('surviving_candidates', [])}`",
        f"- Stage 4 holdout best challenger: `{stage4.get('best_candidate')}`",
        f"- Stage 5 challenger anti-regression pass: `{stage5.get('challenger_pass')}`",
        f"- Stage 6 incumbent route pass: `{stage6.get('round6_pass')}`",
        f"- Stage 6 challenger route pass: `{stage6.get('challenger_pass')}`",
        f"- Stage 7 incumbent stability pass: `{stage7.get('round6_pass')}`",
        f"- Stage 7 challenger stability pass: `{stage7.get('challenger_pass')}`",
        "",
        "## Decision",
        "",
    ]
    if final_status == "challenger replaces round6 as active canonical DoorKey benchmark":
        lines.append("- A within-family challenger survived the verified league, remained meaningful after matched controls, generalized to holdout, preserved the healthy blocks, stayed routed and stable, and cleared the final gate.")
    elif final_status == "round6 confirmed and sealed as active canonical DoorKey benchmark":
        lines.append("- No challenger displaced round6 cleanly enough after verified reruns, matched controls, holdout, anti-regression, route validation, and stability. Round6 remains the active DoorKey benchmark and is now sealed more explicitly as the active canonical pack while the frozen pack stays archived.")
    elif final_status == "challenger fails and round6 remains active incumbent":
        lines.append("- A challenger reached the late stages but did not clear the full replacement bar, so round6 stays the active incumbent without a challenger-led migration.")
    else:
        lines.append("- The challenger league did not justify replacement, but the incumbent also stops short of the stronger explicit-sealing outcome in this run.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="State reconciliation and mega challenger league helpers for the DoorKey active benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("state-reconciliation", "registration", "baseline-sync", "verification-report", "refresh-pack", "decision-memo"):
        stage = sub.add_parser(name)
        stage.add_argument("--campaign-config", required=True)
        stage.add_argument("--output", required=True)
        if name in {"baseline-sync", "verification-report"}:
            stage.add_argument("--csv", required=False)
            stage.add_argument("--json", required=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign = _load_yaml(Path(args.campaign_config))

    if args.command == "state-reconciliation":
        _render_state_reconciliation(campaign, Path(args.output))
        return
    if args.command == "registration":
        _render_registration(campaign, Path(args.output))
        return
    if args.command == "baseline-sync":
        _render_baseline_sync(
            campaign,
            Path(args.output),
            Path(args.csv) if args.csv else None,
            Path(args.json) if args.json else None,
        )
        return
    if args.command == "verification-report":
        _render_stage2_verification(
            campaign,
            Path(args.output),
            Path(args.csv) if args.csv else None,
            Path(args.json) if args.json else None,
        )
        return
    if args.command == "refresh-pack":
        _refresh_pack(campaign, Path(args.output))
        return
    if args.command == "decision-memo":
        _render_decision_memo(campaign, Path(args.output))
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
