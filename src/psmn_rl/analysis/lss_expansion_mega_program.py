from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from psmn_rl.analysis.benchmark_pack import sha256_path
from psmn_rl.analysis.campaign_config import load_campaign_config
from psmn_rl.analysis.lss_post_pass_campaign import (
    DISPLAY_NAMES,
    _float,
    _load_yaml,
    _read_csv_rows,
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


def _winner(
    campaign: dict[str, Any],
    holdout: dict[str, Any],
    anti_regression: dict[str, Any],
    route: dict[str, Any],
    stability: dict[str, Any],
) -> dict[str, Any]:
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
                ("active canonical benchmark within DoorKey only", report_md),
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
        "# Expansion Mega Program State Reconciliation",
        "",
        f"- status: `{status}`",
        f"- active benchmark pack before this program: `{campaign['current_canonical_pack']}`",
        f"- archived frozen pack: `{campaign['legacy_frozen_pack']}`",
        f"- active benchmark candidate before this program: `{campaign['current_canonical_name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Authoritative Current State",
        "",
        "- `round6` is the active DoorKey benchmark before the expansion program starts.",
        "- `outputs/reports/frozen_benchmark_pack.json` remains the archived legacy baseline and provenance anchor.",
        "- The allowed claim envelope remains teacher-guided only, KL learner-state only, DoorKey only, external 64-episode evaluation only, and still excludes PPO-only, specifically multi-expert, cross-task, and KeyCorridor claims.",
        "",
        "## Reconciliation Checks",
        "",
        f"- docs present round6 as active benchmark: `{checks['docs_round6_active']}`",
        f"- claim ledger records the active benchmark state: `{checks['ledger_active_row']}`",
        f"- active benchmark pack names round6: `{checks['pack_active_round6']}`",
        f"- active benchmark pack archives frozen legacy baseline: `{checks['pack_archives_frozen']}`",
        f"- migration memo records active benchmark status: `{checks['migration_memo_active']}`",
        f"- mega-league memo records confirmed active benchmark status: `{checks['mega_memo_active']}`",
        f"- migration gate verdict stays PASS: `{checks['migration_gate_pass']}`",
        f"- mega-league gate verdict stays PASS: `{checks['mega_gate_pass']}`",
        f"- freeze_hardening keeps the active rows on rerender: `{checks['freeze_hardening_tracks_active_rows']}`",
        f"- migration pack source refreshes frozen-pack hashes: `{checks['migration_pack_refreshes_frozen_reference']}`",
        "",
        "## Reconciled Interpretation",
        "",
        "- The repo starts this program from one coherent accepted state: `round6` is already the active DoorKey benchmark, the frozen pack is archived, and both remain comparable through the pack/gate lane.",
        "- This program therefore begins from confirmation-plus-challenger evaluation rather than from an unresolved migration state.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_registration(campaign: dict[str, Any], output: Path) -> None:
    directions = sorted({str(meta["direction"]) for meta in campaign["candidates"].values()})
    exploratory = campaign["blocks"]["exploratory"]
    lines = [
        "# Expansion Mega Program Registration",
        "",
        f"- active incumbent: `{campaign['current_canonical_name']}`",
        f"- active benchmark pack before this program: `{campaign['current_canonical_pack']}`",
        f"- archived frozen pack: `{campaign['legacy_frozen_pack']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Families",
        "",
        f"- DoorKey development families: `{campaign['blocks']['dev']}`",
        f"- DoorKey holdout families: `{campaign['blocks']['holdout']}`",
        f"- DoorKey healthy anti-regression families: `{campaign['blocks']['healthy']}`",
        f"- exploratory adjacent-task track: `{exploratory}`",
        "",
        "## Program Size",
        "",
        f"- mechanism directions: `{len(directions)}`",
        f"- mechanism direction names: `{directions}`",
        f"- challenger variants: `{len(campaign['candidates'])}`",
        "- verification reruns per Stage 1 survivor: `2`",
        "- later-stage evaluation passes for any surviving challenger: `anti-regression + route + stability + exploratory`",
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
        "- If no challenger survives the verified league cleanly enough to replace `round6`, the program still continues through broader DoorKey confirmation plus the fenced exploratory adjacent-task track.",
        "- Stage 8 cannot widen the public claim by itself; it can only classify whether the boundary remains DoorKey-only, has a weak exploratory foothold, or is clearly negative outside DoorKey.",
        "- Any benchmark confirmation or replacement still has to clear the pack/gate lane relative to the archived frozen pack.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_baseline_sync(
    campaign: dict[str, Any],
    output: Path,
    csv_output: Path | None,
    json_output: Path | None,
) -> None:
    frozen_pack = _read_json(Path(campaign["frozen_pack"]))
    active_pack = _read_json(Path(campaign["current_canonical_pack"]))
    current_rows = _current_round6_rows(campaign)
    dev = _control_family(current_rows, str(campaign["current_canonical_name"]), _block_lanes(campaign, "dev"))
    holdout = _control_family(current_rows, str(campaign["current_canonical_name"]), _block_lanes(campaign, "holdout"))
    healthy = _control_family(current_rows, str(campaign["current_canonical_name"]), _block_lanes(campaign, "healthy"))
    lines = [
        "# Expansion Mega Program Baseline Sync",
        "",
        f"- archived frozen pack: `{campaign['frozen_pack']}`",
        f"- active benchmark pack before this program: `{campaign['current_canonical_pack']}`",
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


def _verification_pass(
    campaign: dict[str, Any],
    original: dict[str, Any],
    rerun_a: dict[str, Any],
    rerun_b: dict[str, Any],
    round6_failures: float,
) -> bool:
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


def _render_stage2_verification(
    campaign: dict[str, Any],
    output: Path,
    csv_output: Path | None,
    json_output: Path | None,
) -> None:
    stage1 = _read_json(Path(campaign["reports"]["stage1_raw_json"]))
    rows_a = [
        _candidate_row(record)
        for record in _discover_runs(Path(campaign["stage_roots"]["stage2_verification_a"]))
        if str(record.label) == "kl_lss_sare"
    ]
    rows_b = [
        _candidate_row(record)
        for record in _discover_runs(Path(campaign["stage_roots"]["stage2_verification_b"]))
        if str(record.label) == "kl_lss_sare"
    ]
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
        payload["verified"] = _verification_pass(
            campaign,
            original,
            rerun_a,
            rerun_b,
            _float(round6_summary.get("sare_failures", 0.0)),
        )
        verification_rows.append(payload)
    verified_candidates = [
        row["candidate"]
        for row in sorted(
            [row for row in verification_rows if row["verified"]],
            key=lambda item: ((item["rerun_a_mean"] + item["rerun_b_mean"]) / 2.0, item["original_delta_vs_round6"]),
            reverse=True,
        )
    ]
    effective_stage1 = dict(stage1)
    effective_stage1["advancing_candidates"] = verified_candidates
    effective_stage1["verification_filtered_candidates"] = verified_candidates
    _write_json(Path(campaign["reports"]["stage1_json"]), effective_stage1)
    lines = [
        "# Expansion Mega Program Stage 2 Verification",
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


def _exploratory_candidate_lines(campaign: dict[str, Any]) -> list[str]:
    stage4 = _read_json(Path(campaign["reports"]["stage3_json"])) if Path(campaign["reports"]["stage3_json"]).exists() else {}
    stage5 = _read_json(Path(campaign["reports"]["stage4_json"])) if Path(campaign["reports"]["stage4_json"]).exists() else {}
    stage6 = _read_json(Path(campaign["reports"]["stage5_json"])) if Path(campaign["reports"]["stage5_json"]).exists() else {}
    stage7 = _read_json(Path(campaign["reports"]["stage6_json"])) if Path(campaign["reports"]["stage6_json"]).exists() else {}
    lines = [str(campaign["current_canonical_name"])]
    best_candidate = stage4.get("best_candidate")
    if best_candidate and stage5.get("challenger_pass") and stage6.get("challenger_pass") and stage7.get("challenger_pass"):
        lines.append(str(best_candidate))
    return lines


def _historical_keycorridor_rows() -> list[dict[str, Any]]:
    rows = [
        row
        for row in _read_csv_rows(Path("outputs/reports/lss_keycorridor_transfer_report.csv"))
        if str(row.get("mode")) == "greedy"
    ]
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "line": "historical",
                "candidate": "historical",
                "lane": "keycorridor",
                "seed": int(row["seed"]),
                "label": str(row["label"]),
                "final_greedy_success": _float(row["eval_success_rate"]),
                "run_dir": str(row["run_dir"]),
            }
        )
    return result


def _exploratory_summary(rows: list[dict[str, Any]], line_name: str) -> dict[str, Any]:
    sare_values = [_float(row["final_greedy_success"]) for row in rows if str(row["candidate"]) == line_name and str(row["label"]) == "kl_lss_sare"]
    token_values = [_float(row["final_greedy_success"]) for row in rows if str(row["candidate"]) == line_name and str(row["label"]) == "kl_lss_token_dense"]
    single_values = [_float(row["final_greedy_success"]) for row in rows if str(row["candidate"]) == line_name and str(row["label"]) == "kl_lss_single_expert"]
    sare = _stats(sare_values)
    token = _stats(token_values)
    single = _stats(single_values)
    single_available = bool(single_values)
    control_ceiling = max(token["mean"], single["mean"]) if single_available else token["mean"]
    if sare["mean"] <= 0.05 and sare["complete_seed_failures"] >= max(len(sare_values) - 0.5, 0.0):
        boundary = "clearly negative"
    elif sare["mean"] + 0.02 >= control_ceiling and sare["mean"] >= 0.25:
        boundary = "exploratory foothold only"
    else:
        boundary = "still DoorKey-only"
    return {
        "candidate": line_name,
        "sare_mean": sare["mean"],
        "token_mean": token["mean"],
        "single_mean": single["mean"] if single_available else None,
        "single_available": single_available,
        "sare_failures": sare["complete_seed_failures"],
        "sare_minus_token": sare["mean"] - token["mean"],
        "sare_minus_single": sare["mean"] - single["mean"] if single_available else None,
        "boundary": boundary,
    }


def _overall_exploratory_boundary(summaries: list[dict[str, Any]]) -> str:
    boundaries = {str(row["boundary"]) for row in summaries}
    if "exploratory foothold only" in boundaries:
        return "exploratory foothold only"
    if boundaries == {"clearly negative"}:
        return "clearly negative"
    return "still DoorKey-only"


def _render_exploratory_transfer(
    campaign: dict[str, Any],
    output: Path,
    csv_output: Path | None,
    json_output: Path | None,
) -> None:
    exploratory_root = Path(campaign["stage_roots"]["stage8_exploratory"])
    candidate_lines = _exploratory_candidate_lines(campaign)
    exploratory_rows = [
        _candidate_row(record)
        for record in _discover_runs(exploratory_root)
        if str(record.candidate) in candidate_lines and str(record.label) in {"kl_lss_sare", "kl_lss_token_dense", "kl_lss_single_expert"}
    ]
    historical_rows = _historical_keycorridor_rows()
    summaries = [_exploratory_summary(exploratory_rows, line_name) for line_name in candidate_lines]
    overall_boundary = _overall_exploratory_boundary(summaries)
    recovered_token = _stats([_float(row["final_greedy_success"]) for row in historical_rows if str(row["label"]) == "recovered_token_dense"])
    baseline_sare = _stats([_float(row["final_greedy_success"]) for row in historical_rows if str(row["label"]) == "baseline_sare"])
    lines = [
        "# Expansion Mega Program Stage 8 Exploratory Transfer",
        "",
        f"- exploratory task: `KeyCorridor`",
        f"- evaluated lines: `{candidate_lines}`",
        f"- overall boundary classification: `{overall_boundary}`",
        f"- historical recovered token_dense mean: `{recovered_token['mean']:.4f}`",
        f"- historical baseline PPO SARE mean: `{baseline_sare['mean']:.4f}`",
        "",
        "| Line | KL learner-state SARE | KL learner-state token_dense | KL learner-state single_expert | SARE-token | SARE-single | Failures | Boundary |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        single_mean = f"{_float(row['single_mean']):.4f}" if row.get("single_available") else "n/a"
        sare_minus_single = f"{_float(row['sare_minus_single']):.4f}" if row.get("single_available") else "n/a"
        lines.append(
            f"| `{row['candidate']}` | `{row['sare_mean']:.4f}` | `{row['token_mean']:.4f}` | `{single_mean}` | `{row['sare_minus_token']:.4f}` | `{sare_minus_single}` | `{int(row['sare_failures'])}` | `{row['boundary']}` |"
        )
    lines.extend(
        [
            "",
            "| Line | Seed | Variant | Greedy Success |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for row in sorted(exploratory_rows, key=lambda item: (str(item["candidate"]), int(item["seed"]), str(item["label"]))):
        lines.append(
            f"| `{row['candidate']}` | {row['seed']} | {DISPLAY_NAMES[str(row['label'])]} | {_format_float(_float(row['final_greedy_success']))} |"
        )
    lines.extend(
        [
            "",
            "## Historical Reference",
            "",
            "- `outputs/reports/lss_keycorridor_transfer_report.md` remains the historical bounded negative reference for KeyCorridor and is used here only as exploratory context, not as claim-widening evidence.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, exploratory_rows + historical_rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "candidate_lines": candidate_lines,
                "summaries": summaries,
                "overall_boundary": overall_boundary,
                "historical_recovered_token_mean": recovered_token["mean"],
                "historical_baseline_sare_mean": baseline_sare["mean"],
            },
        )


def _door_key_strengthened(campaign: dict[str, Any], holdout: dict[str, Any], anti_regression: dict[str, Any]) -> bool:
    eps = _float(campaign["selection"]["control_eps"])
    holdout_summary = holdout.get("round6_summary") or {}
    healthy_summary = anti_regression.get("round6_summary") or {}
    if not holdout_summary or not healthy_summary:
        return False
    return (
        _float(holdout_summary.get("sare_mean", 0.0)) + eps >= _float(holdout_summary.get("token_mean", 0.0))
        and _float(holdout_summary.get("sare_mean", 0.0)) + eps >= _float(holdout_summary.get("single_mean", 0.0))
        and _float(healthy_summary.get("sare_mean", 0.0)) + eps >= _float(healthy_summary.get("token_mean", 0.0))
        and _float(healthy_summary.get("sare_mean", 0.0)) + eps >= _float(healthy_summary.get("single_mean", 0.0))
    )


def _decision_status(
    campaign: dict[str, Any],
    holdout: dict[str, Any],
    anti_regression: dict[str, Any],
    route: dict[str, Any],
    stability: dict[str, Any],
    exploratory: dict[str, Any],
    gate_payload: dict[str, Any],
) -> str:
    winner = _winner(campaign, holdout, anti_regression, route, stability)
    gate_verdict = str(gate_payload.get("verdict", gate_payload.get("status", "not_run")))
    required = str(campaign["selection"]["pack_gate_required_verdict"])
    round6_ready = bool(route.get("round6_pass")) and bool(stability.get("round6_pass")) and gate_verdict == required
    if winner["challenger_viable"] and gate_verdict == required:
        return "challenger replaces round6 as active DoorKey benchmark"
    if winner["winner"] == str(campaign["current_canonical_name"]) and round6_ready and _door_key_strengthened(campaign, holdout, anti_regression):
        return "round6 confirmed as active DoorKey benchmark and internal DoorKey envelope strengthened"
    if winner["winner"] == str(campaign["current_canonical_name"]) and round6_ready:
        return "round6 remains active benchmark but envelope does not widen"
    return "benchmark state needs narrowing"


def _refresh_pack(campaign: dict[str, Any], output: Path) -> None:
    pack = _read_json(output)
    holdout = _read_json(Path(campaign["reports"]["stage3_json"])) if Path(campaign["reports"]["stage3_json"]).exists() else {}
    anti_regression = _read_json(Path(campaign["reports"]["stage4_json"])) if Path(campaign["reports"]["stage4_json"]).exists() else {}
    route = _read_json(Path(campaign["reports"]["stage5_json"])) if Path(campaign["reports"]["stage5_json"]).exists() else {}
    stability = _read_json(Path(campaign["reports"]["stage6_json"])) if Path(campaign["reports"]["stage6_json"]).exists() else {}
    exploratory = _read_json(Path(campaign["reports"]["stage8_json"])) if Path(campaign["reports"]["stage8_json"]).exists() else {}
    winner = _winner(campaign, holdout, anti_regression, route, stability)
    pack["frozen_pack_reference"] = {
        "path": str(campaign["frozen_pack"]),
        "sha256": sha256_path(Path(campaign["frozen_pack"])),
        "claim_id": _read_json(Path(campaign["frozen_pack"]))["claim"]["id"],
    }
    pack["expansion_program"] = {
        "winner": winner["winner"],
        "challenger_viable_pre_gate": winner["challenger_viable"],
        "active_pack_before_program": {
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
        "exploratory_transfer": exploratory,
    }
    pack.setdefault("migration", {})
    pack["migration"]["future_comparison_policy"] = {
        "active_canonical_pack": str(output),
        "archived_legacy_pack": str(campaign["legacy_frozen_pack"]),
    }
    pack.setdefault("provenance", {})
    pack["provenance"]["git_commit"] = get_git_commit()
    pack["provenance"]["git_dirty"] = get_git_dirty()
    pack["provenance"]["expansion_program_notes"] = "benchmark expansion + mega challenger + exploratory transfer refresh"
    output.write_text(json.dumps(pack, indent=2, sort_keys=True), encoding="utf-8")


def _render_decision_memo(campaign: dict[str, Any], output: Path) -> None:
    stage1 = _read_json(Path(campaign["reports"]["stage1_raw_json"]))
    stage2 = _read_json(Path(campaign["reports"]["stage2_verification_json"])) if Path(campaign["reports"]["stage2_verification_json"]).exists() else {"verified_candidates": []}
    stage3 = _read_json(Path(campaign["reports"]["stage2_json"])) if Path(campaign["reports"]["stage2_json"]).exists() else {"surviving_candidates": []}
    stage4 = _read_json(Path(campaign["reports"]["stage3_json"])) if Path(campaign["reports"]["stage3_json"]).exists() else {}
    stage5 = _read_json(Path(campaign["reports"]["stage4_json"])) if Path(campaign["reports"]["stage4_json"]).exists() else {}
    stage6 = _read_json(Path(campaign["reports"]["stage5_json"])) if Path(campaign["reports"]["stage5_json"]).exists() else {}
    stage7 = _read_json(Path(campaign["reports"]["stage6_json"])) if Path(campaign["reports"]["stage6_json"]).exists() else {}
    stage8 = _read_json(Path(campaign["reports"]["stage8_json"])) if Path(campaign["reports"]["stage8_json"]).exists() else {}
    gate_payload = _read_json(Path(campaign["reports"]["gate_report_json"])) if Path(campaign["reports"]["gate_report_json"]).exists() else {"status": "not_run"}
    final_status = _decision_status(campaign, stage4, stage5, stage6, stage7, stage8, gate_payload)
    winner = _winner(campaign, stage4, stage5, stage6, stage7)
    lines = [
        "# Expansion Mega Program Decision Memo",
        "",
        f"- final status: `{final_status}`",
        f"- winning line: `{winner['winner']}`",
        f"- gate verdict: `{gate_payload.get('verdict', gate_payload.get('status', 'not_run'))}`",
        f"- exploratory boundary: `{stage8.get('overall_boundary', 'not_run')}`",
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
    if final_status == "challenger replaces round6 as active DoorKey benchmark":
        lines.append("- A within-family challenger survived the 30-run league, stayed meaningful after verification and matched controls, generalized to holdout, preserved the healthy blocks, stayed routed and stable, and cleared the final pack/gate lane strongly enough to replace `round6`.")
    elif final_status == "round6 confirmed as active DoorKey benchmark and internal DoorKey envelope strengthened":
        lines.append("- No challenger displaced `round6` cleanly enough after the larger verified league. `round6` remained control-competitive across the broader DoorKey families, preserved the healthy-block picture, stayed routed and stable, and the exploratory adjacent-task lane still stops short of claim widening, so the internal DoorKey benchmark role is strengthened while the public envelope remains narrow.")
    elif final_status == "round6 remains active benchmark but envelope does not widen":
        lines.append("- No challenger displaced `round6`, and the active benchmark still clears the route/stability/gate bar, but the broader program does not add enough internal strength or exploratory signal to justify stronger state language beyond keeping the public envelope narrow.")
    else:
        lines.append("- The broader program exposed enough weakness that the active benchmark state cannot be strengthened and should be narrowed back operationally.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark expansion and mega challenger helpers for the active DoorKey round6 benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in (
        "state-reconciliation",
        "registration",
        "baseline-sync",
        "verification-report",
        "exploratory-transfer",
        "refresh-pack",
        "decision-memo",
    ):
        stage = sub.add_parser(name)
        stage.add_argument("--campaign-config", required=True)
        stage.add_argument("--output", required=True)
        if name in {"baseline-sync", "verification-report", "exploratory-transfer"}:
            stage.add_argument("--csv", required=False)
            stage.add_argument("--json", required=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign = load_campaign_config(Path(args.campaign_config))

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
    if args.command == "exploratory-transfer":
        _render_exploratory_transfer(
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
