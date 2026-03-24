from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from psmn_rl.analysis.benchmark_pack import sha256_path
from psmn_rl.analysis.campaign_config import load_campaign_config
from psmn_rl.analysis.lss_expansion_mega_program import (
    _door_key_strengthened,
    _exploratory_summary,
    _historical_keycorridor_rows,
    _overall_exploratory_boundary,
    _rerun_summary,
    _verification_pass,
)
from psmn_rl.analysis.lss_post_pass_campaign import (
    DISPLAY_NAMES,
    _float,
    _int,
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
    _stage2_pass,
)
from psmn_rl.utils.io import get_git_commit, get_git_dirty

TRACK_PRIORITY: tuple[str, ...] = ("fruitful", "exploratory", "archpilot")
TRACK_TITLES: dict[str, str] = {
    "fruitful": "Fruitful",
    "exploratory": "Exploratory",
    "archpilot": "Architecture-Adjacent",
}


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _optional_text(path: str | Path | None) -> str:
    if path is None:
        return ""
    target = Path(path)
    if not target.exists():
        return ""
    return target.read_text(encoding="utf-8")


def _optional_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    return _read_json(target)


def _decision_strings(campaign: dict[str, Any]) -> dict[str, str]:
    overrides = campaign.get("decision_strings")
    if isinstance(overrides, dict):
        required = {"replace", "confirm", "narrow", "narrow_state"}
        if required.issubset(overrides):
            return {key: str(overrides[key]) for key in required}
    if bool(campaign.get("generic_decision_language")):
        return {
            "replace": "challenger replaces the active benchmark",
            "confirm": "active benchmark confirmed and internal DoorKey frontier strengthened",
            "narrow": "active benchmark remains and envelope stays narrow",
            "narrow_state": "benchmark/frontier state needs narrowing",
        }
    active_name = str(campaign["current_canonical_name"])
    return {
        "replace": f"challenger replaces {active_name} as active DoorKey benchmark",
        "confirm": f"{active_name} confirmed as active DoorKey benchmark and internal DoorKey benchmark state strengthened",
        "narrow": f"{active_name} remains active benchmark and envelope stays narrow",
        "narrow_state": "benchmark state needs narrowing",
    }


def _candidate_meta(campaign: dict[str, Any], candidate: str) -> dict[str, Any]:
    meta = dict(campaign["candidates"][candidate])
    meta["track"] = str(meta.get("track", "fruitful"))
    meta["family"] = str(meta.get("family", "unassigned"))
    meta["direction"] = str(meta.get("direction", "unspecified"))
    meta["description"] = str(meta.get("description", ""))
    return meta


def _ordered_tracks(tracks: list[str]) -> tuple[str, ...]:
    unique = []
    for track in tracks:
        if track not in unique:
            unique.append(track)
    priority = [track for track in TRACK_PRIORITY if track in unique]
    extras = sorted(track for track in unique if track not in TRACK_PRIORITY)
    return tuple([*priority, *extras])


def _campaign_tracks(campaign: dict[str, Any]) -> tuple[str, ...]:
    tracks = [str(meta.get("track", "fruitful")) for meta in campaign.get("candidates", {}).values()]
    return _ordered_tracks(tracks)


def _summary_tracks(candidate_summaries: list[dict[str, Any]]) -> tuple[str, ...]:
    return _ordered_tracks([str(row.get("track", "fruitful")) for row in candidate_summaries])


def _stage1_top_k(campaign: dict[str, Any], track: str) -> int:
    return _int(campaign["selection"].get(f"stage1_{track}_top_k", 0))


def _stage1_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        _float(row["candidate_mean"]),
        _float(row["delta_vs_round6"]),
        _float(row["candidate_minus_token"]),
        _float(row["candidate_minus_single"]),
    )


def _stage1_candidate_pass(campaign: dict[str, Any], round6: dict[str, Any], row: dict[str, Any]) -> bool:
    if bool(row.get("incomplete")):
        return False
    if _float(row["candidate_failures"]) > _float(round6.get("sare_failures", 0.0)):
        return False
    return _float(row["candidate_mean"]) >= _float(round6.get("sare_mean", 0.0)) + _float(
        campaign["selection"]["min_dev_gain"]
    )


def _stage1_reason(campaign: dict[str, Any], round6: dict[str, Any], row: dict[str, Any], selected: set[str]) -> str:
    if str(row["candidate"]) in selected:
        return "advance"
    if bool(row.get("incomplete")):
        return "stop: catastrophic family prune after incomplete calibration"
    if not bool(row.get("stage1_pass")):
        if _float(row["candidate_failures"]) > _float(round6.get("sare_failures", 0.0)):
            return "stop: new complete-seed failures"
        if _float(row["candidate_mean"]) < _float(round6.get("sare_mean", 0.0)) + _float(campaign["selection"]["min_dev_gain"]):
            return "stop: below incumbent dev mean"
        return "stop: failed stage1 challenger bar"
    track = str(row["track"])
    return f"stop: outside {track} top-{_stage1_top_k(campaign, track)}"


def _selected_stage1_candidates(
    campaign: dict[str, Any],
    candidate_summaries: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], list[str]]:
    selected_by_track: dict[str, list[str]] = {}
    selected: list[str] = []
    for track in _summary_tracks(candidate_summaries):
        top_k = _stage1_top_k(campaign, track)
        chosen = [
            row["candidate"]
            for row in sorted(
                [row for row in candidate_summaries if str(row["track"]) == track and bool(row.get("stage1_pass"))],
                key=_stage1_sort_key,
                reverse=True,
            )[:top_k]
        ]
        selected_by_track[track] = chosen
        selected.extend(chosen)
    return selected_by_track, selected


def _render_state_reconciliation(campaign: dict[str, Any], output: Path) -> None:
    readme = _read_text("README.md")
    report_md = _read_text("report.md")
    summary_md = _read_text("summary.md")
    ledger_md = _read_text("outputs/reports/claim_ledger.md")
    migration_memo = _optional_text("outputs/reports/successor_migration_decision_memo.md")
    mega_memo = _optional_text("outputs/reports/successor_mega_league_decision_memo.md")
    expansion_memo = _optional_text("outputs/reports/expansion_mega_program_decision_memo.md")
    current_decision_memo = _optional_text(campaign.get("current_decision_memo"))
    frontier_manifest = _optional_text(campaign.get("current_frontier_manifest"))
    operational_state = _optional_text(campaign.get("current_operational_state"))
    claim_conformance = _optional_text(campaign.get("claim_gate_conformance_report"))
    active_pack = _read_json(Path(campaign["current_canonical_pack"]))
    active_gate = _optional_json(campaign.get("current_canonical_gate_report"))
    active_name = str(campaign["current_canonical_name"])
    checks = {
        "docs_round6_active": all(
            needle in text
            for needle, text in (
                (active_name, readme),
                ("active DoorKey benchmark", report_md),
                ("active benchmark", summary_md),
            )
        ),
        "ledger_active_row": "active canonical benchmark" in ledger_md or "active DoorKey benchmark" in ledger_md,
        "pack_active_round6": str(active_pack.get("candidate_name")) == active_name,
        "pack_archives_frozen": "archived_legacy_frozen_pack" in active_pack.get("migration", {})
        or "archived_legacy_frozen_pack" in active_pack.get("expansion_program", {}),
        "migration_memo_active": "round6 canonized as active DoorKey benchmark" in migration_memo,
        "mega_memo_active": "round6 confirmed and sealed as active canonical DoorKey benchmark" in mega_memo,
        "expansion_memo_active": "round6 confirmed as active DoorKey benchmark" in expansion_memo,
        "current_decision_active": active_name in current_decision_memo and ("active benchmark" in current_decision_memo or "active DoorKey benchmark" in current_decision_memo),
        "frontier_manifest_active": active_name in frontier_manifest and ("active benchmark" in frontier_manifest or "active DoorKey benchmark" in frontier_manifest),
        "operational_state_active": active_name in operational_state and str(campaign["current_canonical_pack"]) in operational_state,
        "claim_gate_conformance_pass": "PASS" in claim_conformance,
        "active_gate_pass": str(active_gate.get("verdict")) == "PASS: thaw consideration allowed" if active_gate else False,
    }
    status = "reconciled" if all(checks.values()) else "inconsistent"
    lines = [
        "# Portfolio Campaign State Reconciliation",
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
        f"- `{active_name}` is the active DoorKey benchmark before the portfolio program starts.",
        "- `outputs/reports/frozen_benchmark_pack.json` remains the archived legacy baseline and provenance anchor.",
        "- The allowed public claim envelope remains teacher-guided only, KL learner-state only, DoorKey only, and external 64-episode evaluation only.",
        "",
        "## Reconciliation Checks",
        "",
        f"- docs present {active_name} as active benchmark: `{checks['docs_round6_active']}`",
        f"- claim ledger records the active benchmark state: `{checks['ledger_active_row']}`",
        f"- active benchmark pack names {active_name}: `{checks['pack_active_round6']}`",
        f"- active benchmark pack archives frozen legacy baseline: `{checks['pack_archives_frozen']}`",
        f"- migration memo records active benchmark status: `{checks['migration_memo_active']}`",
        f"- mega-league memo records confirmed active benchmark status: `{checks['mega_memo_active']}`",
        f"- expansion-program memo records confirmed active benchmark status: `{checks['expansion_memo_active']}`",
        f"- current decision memo records active benchmark status: `{checks['current_decision_active']}`",
        f"- current frontier manifest records active role ordering: `{checks['frontier_manifest_active']}`",
        f"- current operational state records active pack and benchmark: `{checks['operational_state_active']}`",
        f"- claim-gate conformance report still passes: `{checks['claim_gate_conformance_pass']}`",
        f"- active benchmark gate verdict stays PASS: `{checks['active_gate_pass']}`",
        "",
        "## Reconciled Interpretation",
        "",
        f"- This program starts from one coherent accepted state: `{active_name}` is the active DoorKey benchmark, the frozen pack remains archived, and both stay comparable through the pack/gate lane.",
        "- The portfolio campaign therefore begins from confirmation-plus-challenger evaluation rather than from a pending migration state.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_registration(campaign: dict[str, Any], output: Path) -> None:
    tracks = _campaign_tracks(campaign)
    track_counts = {track: 0 for track in tracks}
    family_counts: dict[str, int] = {}
    directions = sorted({str(meta["direction"]) for meta in campaign["candidates"].values()})
    hard_seed_blocks = list(campaign["blocks"].get("hard_seed", []))
    for candidate in campaign["candidates"]:
        meta = _candidate_meta(campaign, str(candidate))
        track_counts[str(meta["track"])] += 1
        family_counts[str(meta["family"])] = family_counts.get(str(meta["family"]), 0) + 1
    lines = [
        "# Portfolio Campaign Registration",
        "",
        f"- active incumbent: `{campaign['current_canonical_name']}`",
        f"- active benchmark pack before this program: `{campaign['current_canonical_pack']}`",
        f"- archived frozen pack: `{campaign['legacy_frozen_pack']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Track Split",
        "",
    ]
    for track in tracks:
        lines.append(f"- {track} challenger count: `{track_counts[track]}`")
    lines.extend(
        [
            f"- total challenger count: `{len(campaign['candidates'])}`",
            "",
            "## Families",
            "",
        ]
    )
    lines.extend(
        [
        f"- development families: `{campaign['blocks']['dev']}`",
        f"- holdout families: `{campaign['blocks']['holdout']}`",
        f"- healthy anti-regression families: `{campaign['blocks']['healthy']}`",
        f"- hard-seed / hard-pattern families: `{hard_seed_blocks}`",
        f"- exploratory adjacent-task track: `{campaign['blocks']['exploratory']}`",
        f"- family counts: `{family_counts}`",
        "",
        "## Program Size",
        "",
        f"- distinct mechanism directions: `{len(directions)}`",
        f"- mechanism direction names: `{directions}`",
        f"- verification reruns per survivor: `2`",
        f"- route-validation cases: `{campaign.get('route_cases', {})}`",
        f"- stability cases: `{campaign.get('stability_cases', {})}`",
        "",
        "## Fair-Shot Rule",
        "",
        "- Each family gets a bounded calibration sweep before being declared alive or dead.",
        "- A family may be pruned early only if it is catastrophically below `round6` on multiple development families and the rerun confirms the failure.",
        "- No family is promoted on one lucky family group or one lucky seed.",
        "",
        ]
    )
    for track in tracks:
        lines.insert(lines.index("## Fair-Shot Rule"), f"- Stage 1 {track} top-k: `{_stage1_top_k(campaign, track)}`")
    fair_shot_index = lines.index("## Fair-Shot Rule")
    lines.insert(fair_shot_index, "")
    registration_groups = campaign.get("registration_groups", {})
    if registration_groups:
        lines.extend(
            [
                "## Exact Candidate Families",
                "",
            ]
        )
        for title, groups in registration_groups.items():
            lines.append(f"### {title}")
            lines.append("")
            for group in groups:
                label = str(group.get("label", "unlabeled"))
                note = str(group.get("note", ""))
                candidates = [f"`{str(name)}`" for name in group.get("candidates", [])]
                lines.append(f"- `{label}`: {', '.join(candidates)}")
                if note:
                    lines.append(f"  rationale: {note}")
            lines.append("")
    seed_block_rationale = campaign.get("seed_block_rationale", {})
    if seed_block_rationale:
        lines.extend(
            [
                "## Seed-Block Rationale",
                "",
            ]
        )
        for label, note in seed_block_rationale.items():
            lines.append(f"- `{label}`: {note}")
        lines.append("")
    unsupported_directions = list(campaign.get("unsupported_directions", []))
    if unsupported_directions:
        lines.extend(
            [
                "## Explicit Omissions",
                "",
            ]
        )
        for item in unsupported_directions:
            lines.append(f"- {item}")
        lines.append("")
    lines.extend(
        [
        "## Historical Context",
        "",
        ]
    )
    for path in _historical_refs(campaign):
        lines.append(f"- `{path}`")
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
        "# Portfolio Campaign Baseline Sync",
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
        f"- development-family round6 SARE/token/single: `{dev['sare_mean']:.4f}` / `{dev['token_mean']:.4f}` / `{dev['single_mean']:.4f}`",
        f"- holdout-family round6 SARE/token/single: `{holdout['sare_mean']:.4f}` / `{holdout['token_mean']:.4f}` / `{holdout['single_mean']:.4f}`",
        f"- healthy-family round6 SARE/token/single: `{healthy['sare_mean']:.4f}` / `{healthy['token_mean']:.4f}` / `{healthy['single_mean']:.4f}`",
        "",
        "## Current Hard-Seed / Weakness Context",
        "",
    ]
    for path in campaign.get("current_hard_seed_reports", []):
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
        "## Historical Challenger Context",
        "",
        ]
    )
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


def _track_report_json_path(campaign: dict[str, Any], track: str) -> Path | None:
    key = f"stage1_{track}_json"
    if key not in campaign["reports"]:
        return None
    return Path(campaign["reports"][key])


def _track_report_path(campaign: dict[str, Any], track: str) -> Path:
    return Path(campaign["reports"][f"stage1_{track}_report"])


def _render_stage1_track_report(
    campaign: dict[str, Any],
    stage1_payload: dict[str, Any],
    track: str,
) -> None:
    round6 = dict(stage1_payload.get("round6_summary", {}))
    candidate_summaries = [row for row in stage1_payload.get("candidate_summaries", []) if str(row["track"]) == track]
    selected = list(stage1_payload.get(f"{track}_advancing_candidates", []))
    rows = [row for row in stage1_payload.get("rows", []) if str(row.get("track")) == track]
    family_counts: dict[str, int] = {}
    for row in candidate_summaries:
        family = str(row["family"])
        family_counts[family] = family_counts.get(family, 0) + 1
    lines = [
        f"# Portfolio Stage 1 {TRACK_TITLES.get(track, track.replace('_', ' ').title())} Screening",
        "",
        f"- track budget: `{len(candidate_summaries)}` candidates",
        f"- advancing challengers from this track: `{selected}`",
        f"- incumbent round6 dev SARE/token/single: `{_float(round6.get('sare_mean', 0.0)):.4f}` / `{_float(round6.get('token_mean', 0.0)):.4f}` / `{_float(round6.get('single_mean', 0.0)):.4f}`",
        f"- family counts: `{family_counts}`",
        "",
        "| Candidate | Family | Direction | Dev Mean | Δ vs round6 | Candidate-token | Candidate-single | Failures | Observed Specs | Stage 1 | Reason |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in sorted(candidate_summaries, key=_stage1_sort_key, reverse=True):
        lines.append(
            f"| `{row['candidate']}` | `{row['family']}` | `{row['direction']}` | `{_float(row['candidate_mean']):.4f}` | `{_float(row['delta_vs_round6']):.4f}` | `{_float(row['candidate_minus_token']):.4f}` | `{_float(row['candidate_minus_single']):.4f}` | `{int(_float(row['candidate_failures']))}` | `{int(_float(row['observed_dev_specs']))}` / `{int(_float(row['expected_dev_specs']))}` | `{'pass' if row['stage1_pass'] else 'stop'}` | `{row['stage1_reason']}` |"
        )
    lines.extend(["", "| Candidate | Family | Block | Seed | Greedy Success | Δ vs round6 | Δ vs token_dense | Δ vs single_expert |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"])
    for row in sorted(rows, key=lambda item: (str(item["candidate"]), str(item["lane"]), int(item["seed"]))):
        lines.append(
            f"| `{row['candidate']}` | `{row['family']}` | {row['lane']} | {row['seed']} | {_format_float(_float(row['final_greedy_success']))} | {_format_float(_float(row['delta_vs_round6']))} | {_format_float(_float(row['delta_vs_token_dense']))} | {_format_float(_float(row['delta_vs_single_expert']))} |"
        )
    path = _track_report_path(campaign, track)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = _track_report_json_path(campaign, track)
    if json_path is not None:
        _write_json(
            json_path,
            {
                "track": track,
                "advancing_candidates": selected,
                "candidate_summaries": candidate_summaries,
                "rows": rows,
                "round6_summary": round6,
            },
        )


def _render_stage1_screening(campaign: dict[str, Any]) -> None:
    raw_path = Path(campaign["reports"]["stage1_raw_json"])
    effective_path = Path(campaign["reports"]["stage1_json"])
    stage1_payload = _read_json(raw_path)
    round6 = dict(stage1_payload.get("round6_summary", {}))
    expected_dev_specs = sum(len(block.get("seeds", [])) for block in campaign["blocks"]["dev"])
    detail_rows: list[dict[str, Any]] = []
    for row in stage1_payload.get("rows", []):
        payload = dict(row)
        meta = _candidate_meta(campaign, str(payload["candidate"]))
        payload["track"] = meta["track"]
        payload["family"] = meta["family"]
        payload["direction"] = meta["direction"]
        detail_rows.append(payload)
    observed_specs: dict[str, int] = {}
    for row in detail_rows:
        observed_specs[str(row["candidate"])] = observed_specs.get(str(row["candidate"]), 0) + 1
    selected_rows: list[dict[str, Any]] = []
    for row in stage1_payload.get("candidate_summaries", []):
        payload = dict(row)
        meta = _candidate_meta(campaign, str(payload["candidate"]))
        payload.update(meta)
        payload["observed_dev_specs"] = observed_specs.get(str(payload["candidate"]), 0)
        payload["expected_dev_specs"] = expected_dev_specs
        payload["incomplete"] = payload["observed_dev_specs"] < expected_dev_specs
        payload["stage1_pass"] = _stage1_candidate_pass(campaign, round6, payload)
        selected_rows.append(payload)
    selected_by_track, selected = _selected_stage1_candidates(campaign, selected_rows)
    selected_set = set(selected)
    for row in selected_rows:
        row["stage1_reason"] = _stage1_reason(campaign, round6, row, selected_set)
    stage1_payload["raw_advancing_candidates"] = list(stage1_payload.get("advancing_candidates", []))
    stage1_payload["candidate_summaries"] = selected_rows
    stage1_payload["rows"] = detail_rows
    for track in _summary_tracks(selected_rows):
        stage1_payload[f"{track}_advancing_candidates"] = list(selected_by_track.get(track, []))
    stage1_payload["advancing_candidates"] = selected
    stage1_payload["track_candidate_counts"] = {
        track: sum(1 for row in selected_rows if str(row["track"]) == track) for track in _summary_tracks(selected_rows)
    }
    _write_json(raw_path, stage1_payload)
    _write_json(effective_path, stage1_payload)
    for track in _summary_tracks(selected_rows):
        _render_stage1_track_report(campaign, stage1_payload, track)


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
        original = dict(originals[candidate_name])
        rerun_a = _rerun_summary(rows_a, candidate_name)
        rerun_b = _rerun_summary(rows_b, candidate_name)
        payload = {
            "candidate": candidate_name,
            "track": str(original.get("track", "unknown")),
            "family": str(original.get("family", "unknown")),
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
        "# Portfolio Stage 2 Verification",
        "",
        f"- Stage 1 advancing challengers: `{stage1.get('advancing_candidates', [])}`",
        f"- verified challengers: `{verified_candidates}`",
        "",
        "| Candidate | Track | Family | Original Dev Mean | Rerun A | Rerun B | ΔA vs round6 | ΔB vs round6 | Original Failures | Rerun A Failures | Rerun B Failures | Stage 2 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in verification_rows:
        lines.append(
            f"| `{row['candidate']}` | `{row['track']}` | `{row['family']}` | `{row['original_mean']:.4f}` | `{row['rerun_a_mean']:.4f}` | `{row['rerun_b_mean']:.4f}` | `{row['rerun_a_delta_vs_round6']:.4f}` | `{row['rerun_b_delta_vs_round6']:.4f}` | `{int(row['original_failures'])}` | `{int(row['rerun_a_failures'])}` | `{int(row['rerun_b_failures'])}` | `{'pass' if row['verified'] else 'stop'}` |"
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


def _synthetic_control_rows(sare_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sare_rows:
        token_row = dict(row)
        token_row["label"] = "kl_lss_token_dense"
        token_row["final_greedy_success"] = _float(row["final_greedy_success"]) - _float(row["delta_vs_token_dense"])
        token_row["run_dir"] = f"{row['run_dir']}::synthetic_token_dense"
        rows.append(token_row)

        single_row = dict(row)
        single_row["label"] = "kl_lss_single_expert"
        single_row["final_greedy_success"] = _float(row["final_greedy_success"]) - _float(row["delta_vs_single_expert"])
        single_row["run_dir"] = f"{row['run_dir']}::synthetic_single_expert"
        rows.append(single_row)
    return rows


def _render_stage3_fairness(
    campaign: dict[str, Any],
    output: Path,
    csv_output: Path | None,
    json_output: Path | None,
) -> None:
    stage1 = _read_json(Path(campaign["reports"]["stage1_json"]))
    candidate_names = [str(name) for name in stage1.get("advancing_candidates", [])]
    dev_lanes = _block_lanes(campaign, "dev")
    round6_rows = [row for row in _current_round6_rows(campaign, dev_lanes) if str(row["label"]) in CURRENT_LABELS]
    sare_rows = [row for row in stage1.get("rows", []) if str(row["candidate"]) in candidate_names]
    detail_rows = sare_rows + _synthetic_control_rows(sare_rows)
    round6_summary = _control_family(round6_rows, str(campaign["current_canonical_name"]), dev_lanes)

    challenger_summaries: list[dict[str, Any]] = []
    for candidate in candidate_names:
        candidate_rows = [row for row in detail_rows if str(row["candidate"]) == candidate]
        summary = _control_family(candidate_rows, candidate, dev_lanes)
        payload = {
            "candidate": candidate,
            **summary,
            "delta_vs_round6": summary["sare_mean"] - round6_summary["sare_mean"],
        }
        payload["stage2_pass"] = _stage2_pass(campaign, round6_summary, payload)
        challenger_summaries.append(payload)

    surviving_candidates = [
        row["candidate"]
        for row in sorted(
            [row for row in challenger_summaries if row["stage2_pass"]],
            key=lambda item: (item["sare_mean"], item["sare_minus_token"], item["delta_vs_round6"]),
            reverse=True,
        )[: _int(campaign["selection"]["stage2_top_k"])]
    ]

    lines = [
        "# Portfolio Stage 3 Fairness",
        "",
        "- fairness source: `Stage 1 matched-control rows + Stage 2 verified SARE reruns`",
        f"- verified challengers entering fairness: `{candidate_names}`",
        f"- surviving challengers: `{surviving_candidates}`",
        "",
        "| Candidate | Dev SARE | Dev token_dense | Dev single_expert | SARE-token | SARE-single | Δ vs round6 | Failures | Stage 3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| `{campaign['current_canonical_name']}` | `{round6_summary['sare_mean']:.4f}` | `{round6_summary['token_mean']:.4f}` | `{round6_summary['single_mean']:.4f}` | `{round6_summary['sare_minus_token']:.4f}` | `{round6_summary['sare_minus_single']:.4f}` | `0.0000` | `{int(round6_summary['sare_failures'])}` | `incumbent` |",
    ]
    for row in sorted(challenger_summaries, key=lambda item: (item["sare_mean"], item["sare_minus_token"]), reverse=True):
        lines.append(
            f"| `{row['candidate']}` | `{row['sare_mean']:.4f}` | `{row['token_mean']:.4f}` | `{row['single_mean']:.4f}` | `{row['sare_minus_token']:.4f}` | `{row['sare_minus_single']:.4f}` | `{row['delta_vs_round6']:.4f}` | `{int(row['sare_failures'])}` | `{'pass' if row['stage2_pass'] else 'stop'}` |"
        )
    lines.extend(["", "| Candidate | Block | Seed | Variant | Greedy Success |", "| --- | --- | --- | --- | ---: |"])
    for row in sorted(round6_rows + detail_rows, key=lambda item: (str(item["candidate"]), str(item["lane"]), int(item["seed"]), str(item["label"]))):
        if str(row["candidate"]) not in (str(campaign["current_canonical_name"]),) + tuple(candidate_names):
            continue
        if str(row["label"]) not in CURRENT_LABELS:
            continue
        lines.append(
            f"| `{row['candidate']}` | {row['lane']} | {row['seed']} | {DISPLAY_NAMES[str(row['label'])]} | {_format_float(_float(row['final_greedy_success']))} |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, round6_rows + detail_rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "round6_summary": round6_summary,
                "detail_rows": detail_rows,
                "surviving_candidates": surviving_candidates,
                "challenger_summaries": challenger_summaries,
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
        "# Portfolio Stage 8 Exploratory Transfer",
        "",
        "- exploratory task: `KeyCorridor`",
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
    lines.extend(["", "| Line | Seed | Variant | Greedy Success |", "| --- | --- | --- | ---: |"])
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


def _decision_status(
    campaign: dict[str, Any],
    holdout: dict[str, Any],
    anti_regression: dict[str, Any],
    route: dict[str, Any],
    stability: dict[str, Any],
    exploratory: dict[str, Any],
    gate_payload: dict[str, Any],
) -> str:
    labels = _decision_strings(campaign)
    winner = _winner(campaign, holdout, anti_regression, route, stability)
    gate_verdict = str(gate_payload.get("verdict", gate_payload.get("status", "not_run")))
    required = str(campaign["selection"]["pack_gate_required_verdict"])
    round6_ready = bool(route.get("round6_pass")) and bool(stability.get("round6_pass")) and gate_verdict == required
    if winner["challenger_viable"] and gate_verdict == required:
        return labels["replace"]
    if winner["winner"] == str(campaign["current_canonical_name"]) and round6_ready and _door_key_strengthened(campaign, holdout, anti_regression):
        return labels["confirm"]
    if winner["winner"] == str(campaign["current_canonical_name"]) and round6_ready:
        return labels["narrow"]
    return labels["narrow_state"]


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
    pack["portfolio_campaign"] = {
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
    pack["provenance"]["portfolio_campaign_notes"] = "50/50 fruitful vs exploratory benchmark campaign refresh"
    output.write_text(json.dumps(pack, indent=2, sort_keys=True), encoding="utf-8")


def _render_decision_memo(campaign: dict[str, Any], output: Path) -> None:
    stage1 = _read_json(Path(campaign["reports"]["stage1_raw_json"])) if Path(campaign["reports"]["stage1_raw_json"]).exists() else {}
    stage2 = _read_json(Path(campaign["reports"]["stage2_verification_json"])) if Path(campaign["reports"]["stage2_verification_json"]).exists() else {"verified_candidates": []}
    stage3 = _read_json(Path(campaign["reports"]["stage2_json"])) if Path(campaign["reports"]["stage2_json"]).exists() else {"surviving_candidates": []}
    stage4 = _read_json(Path(campaign["reports"]["stage3_json"])) if Path(campaign["reports"]["stage3_json"]).exists() else {}
    stage5 = _read_json(Path(campaign["reports"]["stage4_json"])) if Path(campaign["reports"]["stage4_json"]).exists() else {}
    stage6 = _read_json(Path(campaign["reports"]["stage5_json"])) if Path(campaign["reports"]["stage5_json"]).exists() else {}
    stage7 = _read_json(Path(campaign["reports"]["stage6_json"])) if Path(campaign["reports"]["stage6_json"]).exists() else {}
    stage8 = _read_json(Path(campaign["reports"]["stage8_json"])) if Path(campaign["reports"]["stage8_json"]).exists() else {}
    gate_payload = _read_json(Path(campaign["reports"]["gate_report_json"])) if Path(campaign["reports"]["gate_report_json"]).exists() else {"status": "not_run"}
    final_status = _decision_status(campaign, stage4, stage5, stage6, stage7, stage8, gate_payload)
    labels = _decision_strings(campaign)
    winner = _winner(campaign, stage4, stage5, stage6, stage7)
    lines = [
        "# Portfolio Campaign Decision Memo",
        "",
        f"- final status: `{final_status}`",
        f"- winning line: `{winner['winner']}`",
        f"- gate verdict: `{gate_payload.get('verdict', gate_payload.get('status', 'not_run'))}`",
        f"- exploratory boundary: `{stage8.get('overall_boundary', 'not_run')}`",
        "",
        "## Portfolio Funnel",
        "",
    ]
    for track in _campaign_tracks(campaign):
        lines.append(f"- Stage 1 {track} advancing challengers: `{stage1.get(f'{track}_advancing_candidates', [])}`")
    lines.extend(
        [
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
    )
    if final_status == labels["replace"]:
        lines.append("- A within-family challenger survived the balanced portfolio, stayed meaningful after verification and matched controls, generalized to holdout, preserved the healthy blocks, stayed routed and stable, and cleared the final pack/gate lane strongly enough to replace `round6`.")
    elif final_status == labels["confirm"]:
        lines.append("- No challenger displaced `round6` cleanly enough after the broader 50/50 portfolio. `round6` remained control-competitive across the broader DoorKey families, preserved the healthy-block picture, stayed routed and stable, and the exploratory adjacent-task lane still stops short of claim widening, so the internal DoorKey benchmark role is strengthened while the public envelope remains narrow.")
    elif final_status == labels["narrow"]:
        lines.append("- No challenger displaced `round6`, and the active benchmark still clears the route/stability/gate bar, but the broader program does not add enough internal strength or exploratory signal to justify stronger state language beyond keeping the public envelope narrow.")
    else:
        lines.append("- The broader program exposed enough weakness that the active benchmark state cannot be strengthened and should be narrowed back operationally.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="50/50 fruitful vs exploratory portfolio helpers for the active DoorKey round6 benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in (
        "state-reconciliation",
        "registration",
        "baseline-sync",
        "stage1-screening",
        "verification-report",
        "fairness-report",
        "exploratory-transfer",
        "refresh-pack",
        "decision-memo",
    ):
        stage = sub.add_parser(name)
        stage.add_argument("--campaign-config", required=True)
        if name != "stage1-screening":
            stage.add_argument("--output", required=True)
        if name in {"baseline-sync", "verification-report", "fairness-report", "exploratory-transfer"}:
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
    if args.command == "stage1-screening":
        _render_stage1_screening(campaign)
        return
    if args.command == "verification-report":
        _render_stage2_verification(
            campaign,
            Path(args.output),
            Path(args.csv) if args.csv else None,
            Path(args.json) if args.json else None,
        )
        return
    if args.command == "fairness-report":
        _render_stage3_fairness(
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
