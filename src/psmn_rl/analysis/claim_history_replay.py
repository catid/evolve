from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from psmn_rl.analysis.benchmark_pack import (
    CANDIDATE_PACK_TYPE,
    load_structured_file,
    safe_load_structured_file,
    sha256_path,
    validate_frozen_benchmark_pack,
)
from psmn_rl.analysis.claim_gate import evaluate_pack_claim_gate_from_paths, render_pack_gate_report
from psmn_rl.analysis.freeze_hardening import LEDGER_ROWS
from psmn_rl.utils.io import get_git_commit, get_git_dirty

REPORTS_DIR = Path("outputs/reports")
EPSILON = 1e-9


@dataclass(slots=True)
class HistoricalCaseResult:
    case_id: str
    family: str
    ledger_family: str
    phase_status: str
    description: str
    expected_verdict: str
    actual_verdict: str
    verdict_match: bool
    rationale: str
    pack_path: str
    gate_report: str
    gate_json: str
    reason_summary: str


@dataclass(slots=True)
class LedgerAuditRow:
    family: str
    ledger_status: str
    mapped_cases: str
    replay_verdicts: str
    classification: str
    detail: str


def _as_mapping(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _as_list(value: Any) -> list[Any] | None:
    return value if isinstance(value, list) else None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = float(value)
        except ValueError:
            return None
        if parsed.is_integer():
            return int(parsed)
    return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_matches(row: dict[str, str], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = row.get(key, "")
        if isinstance(expected, list):
            if actual not in [str(item) for item in expected]:
                return False
            continue
        if actual != str(expected):
            return False
    return True


def _actual_set_from_rows(rows: list[dict[str, str]], source: dict[str, Any]) -> set[tuple[str, int]]:
    actual_set: set[tuple[str, int]] = set()
    lane_field = source.get("lane_field")
    seed_field = source.get("seed_field")
    default_lane = source.get("default_lane")
    if seed_field is None:
        return actual_set
    for row in rows:
        seed_value = _coerce_int(row.get(str(seed_field), ""))
        if seed_value is None:
            continue
        if lane_field is not None:
            lane = row.get(str(lane_field), "")
        else:
            lane = ""
        if not lane:
            lane = str(default_lane or "")
        if lane:
            actual_set.add((lane, seed_value))
    return actual_set


def _grouped_metric_values(source: dict[str, Any]) -> tuple[dict[str, list[float]], set[tuple[str, int]]]:
    path = Path(str(source["path"]))
    rows = _read_csv_rows(path)
    raw_filters = _as_mapping(source.get("filters")) or {}
    filters = {str(key): value for key, value in raw_filters.items()}
    filtered = [row for row in rows if _row_matches(row, filters)]
    group_field = str(source["group_field"])
    value_field = str(source["value_field"])
    group_mapping = {str(key): str(value) for key, value in (_as_mapping(source.get("group_mapping")) or {}).items()}
    values: dict[str, list[float]] = {}
    for row in filtered:
        raw_key = row.get(group_field, "")
        metric_key = group_mapping.get(raw_key, raw_key)
        if not metric_key:
            continue
        metric_value = _coerce_float(row.get(value_field, ""))
        if metric_value is None:
            continue
        values.setdefault(metric_key, []).append(metric_value)
    return values, _actual_set_from_rows(filtered, source)


def _merge_values(target: dict[str, list[float]], source_values: dict[str, list[float]]) -> None:
    for key, values in source_values.items():
        target.setdefault(key, []).extend(values)


def _metric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "complete_seed_failures": sum(1 for value in values if abs(value) <= EPSILON),
        "seed_count": len(values),
    }


def _normalize_manual_metrics(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for key, payload in raw.items():
        metric_payload = _as_mapping(payload)
        if metric_payload is None:
            continue
        normalized[str(key)] = {
            "mean": float(metric_payload["mean"]),
            "min": float(metric_payload["min"]),
            "max": float(metric_payload["max"]),
            "complete_seed_failures": int(metric_payload["complete_seed_failures"]),
            "seed_count": int(metric_payload["seed_count"]),
        }
    return normalized


def _build_metric_block(block_spec: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], set[tuple[str, int]]]:
    aggregate_values: dict[str, list[float]] = {}
    actual_set: set[tuple[str, int]] = set()
    for source in _as_list(block_spec.get("sources")) or []:
        source_mapping = _as_mapping(source)
        if source_mapping is None:
            continue
        kind = str(source_mapping.get("kind", ""))
        if kind != "grouped_csv":
            raise ValueError(f"Unsupported historical metric source kind: {kind}")
        source_values, source_set = _grouped_metric_values(source_mapping)
        _merge_values(aggregate_values, source_values)
        actual_set |= source_set
    metrics = {key: _metric_summary(values) for key, values in aggregate_values.items()}
    for key, payload in _normalize_manual_metrics(_as_mapping(block_spec.get("manual_metrics")) or {}).items():
        metrics[key] = payload
    override = _as_list(block_spec.get("actual_sets_override"))
    if override is not None:
        actual_set = {
            (str(item[0]), int(item[1]))
            for item in override
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str) and _coerce_int(item[1]) is not None
        }
    return metrics, actual_set


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_rows(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, payload in sorted(metrics.items()):
        rows.append(
            {
                "variant": variant,
                "mean": f"{float(payload['mean']):.4f}",
                "min": f"{float(payload['min']):.4f}",
                "max": f"{float(payload['max']):.4f}",
                "complete_seed_failures": int(payload["complete_seed_failures"]),
                "seed_count": int(payload["seed_count"]),
            }
        )
    return rows


def _render_metric_report(title: str, metrics: dict[str, dict[str, Any]], actual_set: set[tuple[str, int]]) -> str:
    lines = [
        f"# {title}",
        "",
        f"- lane/seed coverage: `{sorted(actual_set)}`",
        "",
        "| Variant | Mean | Min | Max | Complete Seed Failures | Seed Count |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _metric_rows(metrics):
        lines.append(
            f"| `{row['variant']}` | `{row['mean']}` | `{row['min']}` | `{row['max']}` | `{row['complete_seed_failures']}` | `{row['seed_count']}` |"
        )
    if not metrics:
        lines.append("| `-` | `-` | `-` | `-` | `-` | `-` |")
    return "\n".join(lines) + "\n"


def _candidate_metrics_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": candidate.get("schema_version"),
        "candidate_name": candidate.get("candidate_name"),
        "task": candidate.get("task"),
        "evaluation": candidate.get("evaluation"),
        "requested_claims": candidate.get("requested_claims"),
        "controls_present": candidate.get("controls_present"),
        "metrics": candidate.get("metrics"),
        "actual_sets": candidate.get("actual_sets"),
        "provenance": candidate.get("provenance"),
    }


def _artifact_record(role: str, path: Path) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def _build_historical_candidate_pack(
    frozen_pack_path: Path,
    frozen_pack: dict[str, Any],
    case: dict[str, Any],
    output_dir: Path,
) -> Path:
    case_id = str(case["id"])
    case_dir = output_dir / case_id
    artifact_dir = case_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    combined_metrics, combined_set = _build_metric_block(_as_mapping(case.get("combined_metrics")) or {})
    retry_metrics, retry_set = _build_metric_block(_as_mapping(case.get("retry_block_metrics")) or {})
    evaluation = _as_mapping(case.get("evaluation")) or {}

    candidate: dict[str, Any] = {
        "schema_version": int(frozen_pack.get("candidate_pack", {}).get("schema_version", 1)),
        "pack_type": CANDIDATE_PACK_TYPE,
        "candidate_name": case_id,
        "frozen_pack_reference": {
            "path": str(frozen_pack_path),
            "sha256": sha256_path(frozen_pack_path),
            "claim_id": frozen_pack.get("claim", {}).get("id", ""),
        },
        "task": str(case.get("task", frozen_pack.get("evaluation", {}).get("task", "DoorKey"))),
        "evaluation": {
            "path_key": str(evaluation.get("path_key", frozen_pack.get("evaluation", {}).get("path_key", "external_policy_diagnostics"))),
            "episodes": int(evaluation.get("episodes", frozen_pack.get("evaluation", {}).get("episodes", 64))),
        },
        "requested_claims": [str(item) for item in (_as_list(case.get("requested_claims")) or [])],
        "controls_present": [str(item) for item in (_as_list(case.get("controls_present")) or [])],
        "metrics": {
            "combined": combined_metrics,
            "retry_block": retry_metrics,
        },
        "actual_sets": {
            "combined_lane_seeds": [[lane, seed] for lane, seed in sorted(combined_set)],
            "retry_block_lane_seeds": [[lane, seed] for lane, seed in sorted(retry_set)],
        },
        "provenance": {
            "git_commit": get_git_commit(),
            "git_dirty": get_git_dirty(),
            "historical_phase": str(case.get("description", case_id)),
            "historical_status": str(case.get("phase_status", "")),
            "source_reports": [str(item) for item in (_as_list(case.get("source_reports")) or [])],
            "provenance_note": str(case.get("provenance_note", "")),
        },
        "historical_context": {
            "family": str(case.get("family", "")),
            "ledger_family": str(case.get("ledger_family", "")),
            "description": str(case.get("description", "")),
            "expected_verdict": str(case.get("expected_verdict", "")),
            "rationale": str(case.get("rationale", "")),
        },
    }

    candidate_summary_path = artifact_dir / "candidate_summary.md"
    combined_md_path = artifact_dir / "combined_report.md"
    combined_csv_path = artifact_dir / "combined_report.csv"
    retry_md_path = artifact_dir / "retry_block_report.md"
    retry_csv_path = artifact_dir / "retry_block_report.csv"
    metrics_json_path = artifact_dir / "candidate_metrics.json"

    candidate_summary_lines = [
        f"# Historical Candidate Pack: {case_id}",
        "",
        f"- family: `{case.get('family', '-')}`",
        f"- ledger family: `{case.get('ledger_family', '-')}`",
        f"- historical status: `{case.get('phase_status', '-')}`",
        f"- expected verdict under current gate: `{case.get('expected_verdict', '-')}`",
        f"- rationale: {case.get('rationale', '-')}",
        f"- evaluation: `{candidate['evaluation']['path_key']}` / `{candidate['evaluation']['episodes']}` episodes",
        f"- requested claims: `{candidate['requested_claims']}`",
        f"- controls present: `{candidate['controls_present']}`",
        "",
        "## Source Reports",
        "",
    ]
    for report in _as_list(case.get("source_reports")) or []:
        candidate_summary_lines.append(f"- `{report}`")
    if case.get("provenance_note"):
        candidate_summary_lines.extend(["", "## Provenance Note", "", str(case["provenance_note"])])
    candidate_summary_path.write_text("\n".join(candidate_summary_lines) + "\n", encoding="utf-8")

    combined_md_path.write_text(
        _render_metric_report(f"{case_id} Combined Adapter Report", combined_metrics, combined_set),
        encoding="utf-8",
    )
    combined_csv_rows = _metric_rows(combined_metrics)
    if not combined_csv_rows:
        combined_csv_rows = [{"variant": "-", "mean": "-", "min": "-", "max": "-", "complete_seed_failures": "-", "seed_count": "-"}]
    _write_csv(
        combined_csv_path,
        ["variant", "mean", "min", "max", "complete_seed_failures", "seed_count"],
        combined_csv_rows,
    )
    retry_md_path.write_text(
        _render_metric_report(f"{case_id} Retry-Block Adapter Report", retry_metrics, retry_set),
        encoding="utf-8",
    )
    retry_csv_rows = _metric_rows(retry_metrics)
    if not retry_csv_rows:
        retry_csv_rows = [{"variant": "-", "mean": "-", "min": "-", "max": "-", "complete_seed_failures": "-", "seed_count": "-"}]
    _write_csv(
        retry_csv_path,
        ["variant", "mean", "min", "max", "complete_seed_failures", "seed_count"],
        retry_csv_rows,
    )

    _write_json(metrics_json_path, _candidate_metrics_payload(candidate))
    candidate["artifacts"] = [
        _artifact_record("candidate_summary_markdown", candidate_summary_path),
        _artifact_record("candidate_metrics_json", metrics_json_path),
        _artifact_record("combined_report_markdown", combined_md_path),
        _artifact_record("combined_report_csv", combined_csv_path),
        _artifact_record("retry_block_report_markdown", retry_md_path),
        _artifact_record("retry_block_report_csv", retry_csv_path),
    ]
    _write_json(metrics_json_path, _candidate_metrics_payload(candidate))
    candidate["artifacts"][1] = _artifact_record("candidate_metrics_json", metrics_json_path)

    pack_path = case_dir / f"{case_id}.json"
    _write_json(pack_path, candidate)
    return pack_path


def _render_catalog_report(catalog_path: Path, cases: list[dict[str, Any]], output_dir: Path) -> str:
    lines = [
        "# Historical Candidate Pack Catalog",
        "",
        f"- catalog: `{catalog_path}`",
        f"- output directory: `{output_dir}`",
        "",
        "| Case | Family | Ledger Family | Expected Verdict | Rationale | Candidate Pack |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        case_id = str(case["id"])
        lines.append(
            f"| `{case_id}` | {case.get('family', '-')} | {case.get('ledger_family', '-')} | `{case.get('expected_verdict', '-')}` | {case.get('rationale', '-')} | `{output_dir / case_id / f'{case_id}.json'}` |"
        )
    return "\n".join(lines) + "\n"


def _reason_summary(checks: list[dict[str, str]]) -> str:
    notable = [f"{check['name']}={check['status']}" for check in checks if check["status"] != "PASS"]
    if not notable:
        return "all checks passed"
    return ", ".join(notable[:5])


def _render_replay_report(results: list[HistoricalCaseResult], frozen_pack_path: Path) -> str:
    matched = sum(1 for result in results if result.verdict_match)
    lines = [
        "# Claim History Replay Report",
        "",
        f"- frozen benchmark pack: `{frozen_pack_path}`",
        f"- replayed historical packs: `{len(results)}`",
        f"- expected/actual matches: `{matched}/{len(results)}`",
        "",
        "| Case | Family | Expected | Actual | Match | Reason Summary | Gate Report |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| `{result.case_id}` | {result.family} | `{result.expected_verdict}` | `{result.actual_verdict}` | `{'PASS' if result.verdict_match else 'FAIL'}` | {result.reason_summary} | `{result.gate_report}` |"
        )
    verdict = "PASS: historical replay matches the expected claim-gate verdict map" if matched == len(results) else "FAIL: historical replay mismatches detected"
    lines.extend(["", "## Verdict", "", verdict])
    return "\n".join(lines) + "\n"


def _render_discrepancies_report(
    results: list[HistoricalCaseResult],
    audit_rows: list[LedgerAuditRow],
    ledger_touches: list[str],
) -> str:
    lines = [
        "# Claim History Replay Discrepancies",
        "",
    ]
    replay_mismatches = [result for result in results if not result.verdict_match]
    audit_mismatches = [row for row in audit_rows if row.classification == "inconsistent and needs correction"]
    if not replay_mismatches and not audit_mismatches and not ledger_touches:
        lines.extend(
            [
                "No unresolved replay discrepancies remain.",
                "",
                "- all replay verdicts matched expectations",
                "- no ledger row required correction beyond the replay-backed interpretation now captured in the audit",
            ]
        )
        return "\n".join(lines) + "\n"

    if replay_mismatches:
        lines.extend(["## Replay Verdict Mismatches", ""])
        for result in replay_mismatches:
            lines.append(
                f"- `{result.case_id}` expected `{result.expected_verdict}` but got `{result.actual_verdict}`. See `{result.gate_report}`."
            )
    if audit_mismatches:
        lines.extend(["", "## Ledger Audit Mismatches", ""])
        for row in audit_mismatches:
            lines.append(f"- `{row.family}`: {row.detail}")
    if ledger_touches:
        lines.extend(["", "## Ledger Wording Updates", ""])
        for item in ledger_touches:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _classify_ledger_row(status: str, verdicts: set[str], all_expected_match: bool) -> tuple[str, str]:
    if not all_expected_match:
        return "inconsistent and needs correction", "replay verdicts diverged from the declared expected verdicts"
    if status == "frozen":
        if verdicts == {"FAIL: claim remains frozen"}:
            return "consistent", "frozen rows still replay to a frozen verdict"
        return "inconsistent and needs correction", "frozen row did not replay to a frozen verdict"
    if status == "negative":
        if verdicts == {"FAIL: claim remains frozen"}:
            return "consistent", "negative row replays to a hard gate failure"
        if verdicts <= {"FAIL: claim remains frozen", "INCONCLUSIVE: missing prerequisites"}:
            return "consistent but narrower under current gate", "negative row stays blocked, but current gate treats it as not fully comparable rather than directly claim-failing"
        return "inconsistent and needs correction", "negative row produced an unexpected gate verdict"
    if status == "bounded positive":
        if verdicts <= {"FAIL: claim remains frozen", "INCONCLUSIVE: missing prerequisites"}:
            return "consistent but narrower under current gate", "historical positive signal still does not clear the modern frozen gate"
        return "inconsistent and needs correction", "bounded-positive row produced an unexpected gate verdict"
    return "cannot be evaluated due to missing prerequisites", "ledger status is not recognized by the replay classifier"


def _render_ledger_consistency_report(audit_rows: list[LedgerAuditRow]) -> str:
    lines = [
        "# Claim Ledger Consistency Audit",
        "",
        "| Ledger Family | Ledger Status | Historical Packs | Replay Verdicts | Classification | Detail |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in audit_rows:
        lines.append(
            f"| {row.family} | `{row.ledger_status}` | {row.mapped_cases} | {row.replay_verdicts} | `{row.classification}` | {row.detail} |"
        )
    return "\n".join(lines) + "\n"


def _render_snapshot_report(
    golden_path: Path,
    actual_path: Path,
    snapshot_matches: bool,
    mismatch_count: int,
) -> str:
    verdict = "PASS: historical replay snapshot matches the accepted gate history" if snapshot_matches else "FAIL: historical replay snapshot drift detected"
    lines = [
        "# Claim History Replay Snapshot Report",
        "",
        f"- golden snapshot: `{golden_path}`",
        f"- actual snapshot: `{actual_path}`",
        f"- mismatches: `{mismatch_count}`",
        "",
        "## Verdict",
        "",
        verdict,
    ]
    return "\n".join(lines) + "\n"


def _render_ci_report(workflow_path: Path, replay_json: dict[str, Any]) -> str:
    lines = [
        "# CI Claim History Replay Report",
        "",
        f"- workflow: `{workflow_path}`",
        f"- replay verdict: `{replay_json['verdict']}`",
        f"- snapshot status: `{'PASS' if replay_json['snapshot_matches'] else 'FAIL'}`",
        f"- historical cases covered: `{len(replay_json['cases'])}`",
        "",
        "## Workflow Coverage",
        "",
        "- validates the frozen benchmark pack",
        "- runs the adversarial claim-gate conformance corpus",
        "- replays the gate against the real historical candidate-pack catalog",
        "- checks the historical replay golden snapshot for drift",
        "- runs the full test suite",
    ]
    return "\n".join(lines) + "\n"


def _render_decision_memo(replay_json: dict[str, Any], audit_rows: list[LedgerAuditRow]) -> str:
    pass_count = sum(1 for case in replay_json["cases"] if case["actual_verdict"] == "PASS: thaw consideration allowed")
    fail_count = sum(1 for case in replay_json["cases"] if case["actual_verdict"] == "FAIL: claim remains frozen")
    inconclusive_count = sum(1 for case in replay_json["cases"] if case["actual_verdict"] == "INCONCLUSIVE: missing prerequisites")
    consistent_rows = sum(1 for row in audit_rows if row.classification == "consistent")
    narrower_rows = sum(1 for row in audit_rows if row.classification == "consistent but narrower under current gate")
    inconsistent_rows = sum(1 for row in audit_rows if row.classification == "inconsistent and needs correction")
    lines = [
        "# Claim-History Replay Decision Memo",
        "",
        "## Replay Outcome",
        "",
        f"- replay cases: `{len(replay_json['cases'])}`",
        f"- `PASS` verdicts on historical cases: `{pass_count}`",
        f"- `FAIL` verdicts on historical cases: `{fail_count}`",
        f"- `INCONCLUSIVE` verdicts on historical cases: `{inconclusive_count}`",
        "",
        "## Gate vs History",
        "",
        "- The current gate blocks the PPO-only negative family and the later over-broad multi-expert wording instead of letting them drift into thaw consideration.",
        "- Early positive learner-state phases replay as missing-prerequisite or narrower cases rather than modern thaw candidates because they lack the final fairness controls and retry-block slice.",
        "- The final frozen-era phases replay to the same frozen verdict the repo currently accepts.",
        "",
        "## Ledger Alignment",
        "",
        f"- consistent ledger rows: `{consistent_rows}`",
        f"- consistent-but-narrower rows: `{narrower_rows}`",
        f"- inconsistent rows: `{inconsistent_rows}`",
        "",
        "## Final Result",
        "",
        f"- replay verdict map: `{replay_json['verdict']}`",
        f"- snapshot status: `{'PASS' if replay_json['snapshot_matches'] else 'FAIL'}`",
        "- operational state: frozen DoorKey claim remains sealed, and the gate is now checked against both adversarial packs and the real historical claim trajectory.",
    ]
    return "\n".join(lines) + "\n"


def _run_ledger_consistency_audit(
    cases: list[dict[str, Any]],
    results: list[HistoricalCaseResult],
) -> tuple[list[LedgerAuditRow], list[str]]:
    case_by_id = {result.case_id: result for result in results}
    case_families: dict[str, list[str]] = {}
    for case in cases:
        case_families.setdefault(str(case.get("ledger_family", "")), []).append(str(case["id"]))

    ledger_touches: list[str] = []
    audit_rows: list[LedgerAuditRow] = []
    for row in LEDGER_ROWS:
        family = str(row["family"])
        mapped = case_families.get(family, [])
        if not mapped:
            audit_rows.append(
                LedgerAuditRow(
                    family=family,
                    ledger_status=str(row["status"]),
                    mapped_cases="-",
                    replay_verdicts="-",
                    classification="cannot be evaluated due to missing prerequisites",
                    detail="no historical replay pack was mapped to this ledger row",
                )
            )
            continue
        mapped_results = [case_by_id[item] for item in mapped]
        verdicts = {result.actual_verdict for result in mapped_results}
        all_expected_match = all(result.verdict_match for result in mapped_results)
        classification, detail = _classify_ledger_row(str(row["status"]), verdicts, all_expected_match)
        if family == "Claim broadening" and classification == "consistent but narrower under current gate":
            ledger_touches.append(
                "The claim-broadening row is intentionally interpreted as a temporary within-DoorKey strengthening that the current gate would still block because the historical phase requested now-disallowed multi-expert claim language."
            )
        if family == "Learner-state supervision" and classification == "consistent but narrower under current gate":
            ledger_touches.append(
                "The learner-state supervision row remains a bounded positive signal, but historical replay now makes explicit that the early positive lane does not satisfy the modern frozen-gate prerequisites."
            )
        audit_rows.append(
            LedgerAuditRow(
                family=family,
                ledger_status=str(row["status"]),
                mapped_cases=", ".join(f"`{item}`" for item in mapped),
                replay_verdicts=", ".join(f"`{result.actual_verdict}`" for result in mapped_results),
                classification=classification,
                detail=detail,
            )
        )
    return audit_rows, ledger_touches


def run_history_replay(
    frozen_pack_path: Path,
    catalog_path: Path,
    golden_snapshot_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_pack = load_structured_file(frozen_pack_path)
    frozen_verdict, _frozen_checks = validate_frozen_benchmark_pack(frozen_pack)
    if frozen_verdict != "PASS: frozen benchmark pack validated":
        raise SystemExit(frozen_verdict)

    catalog = load_structured_file(catalog_path)
    cases = [case for case in (_as_list(catalog.get("cases")) or []) if _as_mapping(case) is not None]
    packs_dir = output_dir
    gate_reports_dir = output_dir / "gate_reports"
    gate_reports_dir.mkdir(parents=True, exist_ok=True)

    results: list[HistoricalCaseResult] = []
    for raw_case in cases:
        case = dict(raw_case)
        pack_path = _build_historical_candidate_pack(frozen_pack_path, frozen_pack, case, packs_dir)
        verdict, checks, frozen_rows, candidate_rows = evaluate_pack_claim_gate_from_paths(frozen_pack_path, pack_path)
        gate_report_path = gate_reports_dir / f"{case['id']}.md"
        gate_json_path = gate_reports_dir / f"{case['id']}.json"
        gate_report_path.write_text(
            render_pack_gate_report(
                frozen_pack_path,
                pack_path,
                verdict,
                frozen_rows,
                candidate_rows,
                checks,
            ),
            encoding="utf-8",
        )
        _write_json(
            gate_json_path,
            {
                "frozen_pack": str(frozen_pack_path),
                "candidate_pack": str(pack_path),
                "verdict": verdict,
                "frozen_pack_validation": frozen_rows,
                "candidate_pack_validation": candidate_rows,
                "checks": [asdict(check) for check in checks],
            },
        )
        results.append(
            HistoricalCaseResult(
                case_id=str(case["id"]),
                family=str(case.get("family", "")),
                ledger_family=str(case.get("ledger_family", "")),
                phase_status=str(case.get("phase_status", "")),
                description=str(case.get("description", "")),
                expected_verdict=str(case.get("expected_verdict", "")),
                actual_verdict=verdict,
                verdict_match=verdict == str(case.get("expected_verdict", "")),
                rationale=str(case.get("rationale", "")),
                pack_path=str(pack_path),
                gate_report=str(gate_report_path),
                gate_json=str(gate_json_path),
                reason_summary=_reason_summary([asdict(check) for check in checks]),
            )
        )

    audit_rows, ledger_touches = _run_ledger_consistency_audit(cases, results)

    catalog_report_path = REPORTS_DIR / "historical_candidate_pack_catalog.md"
    replay_report_path = REPORTS_DIR / "claim_history_replay_report.md"
    replay_csv_path = REPORTS_DIR / "claim_history_replay_report.csv"
    replay_json_path = REPORTS_DIR / "claim_history_replay_report.json"
    audit_report_path = REPORTS_DIR / "claim_ledger_consistency_audit.md"
    discrepancy_report_path = REPORTS_DIR / "claim_history_replay_discrepancies.md"
    snapshot_report_path = REPORTS_DIR / "claim_history_replay_snapshot_report.md"
    ci_report_path = REPORTS_DIR / "ci_claim_history_replay_report.md"
    decision_memo_path = REPORTS_DIR / "claim_history_replay_decision_memo.md"
    actual_snapshot_path = output_dir / "claim_history_replay_snapshot_actual.json"

    catalog_report_path.write_text(_render_catalog_report(catalog_path, cases, packs_dir), encoding="utf-8")
    replay_report_path.write_text(_render_replay_report(results, frozen_pack_path), encoding="utf-8")
    _write_csv(
        replay_csv_path,
        [
            "case_id",
            "family",
            "ledger_family",
            "phase_status",
            "expected_verdict",
            "actual_verdict",
            "verdict_match",
            "reason_summary",
            "pack_path",
            "gate_report",
        ],
        [
            {
                "case_id": result.case_id,
                "family": result.family,
                "ledger_family": result.ledger_family,
                "phase_status": result.phase_status,
                "expected_verdict": result.expected_verdict,
                "actual_verdict": result.actual_verdict,
                "verdict_match": result.verdict_match,
                "reason_summary": result.reason_summary,
                "pack_path": result.pack_path,
                "gate_report": result.gate_report,
            }
            for result in results
        ],
    )

    replay_verdict = (
        "PASS: historical replay matches the expected claim-gate verdict map"
        if all(result.verdict_match for result in results)
        else "FAIL: historical replay mismatches detected"
    )
    replay_json = {
        "schema_version": 1,
        "frozen_pack": str(frozen_pack_path),
        "verdict": replay_verdict,
        "cases": [asdict(result) for result in results],
    }
    _write_json(replay_json_path, replay_json)

    audit_report_path.write_text(_render_ledger_consistency_report(audit_rows), encoding="utf-8")
    discrepancy_report_path.write_text(
        _render_discrepancies_report(results, audit_rows, ledger_touches),
        encoding="utf-8",
    )

    snapshot = {
        "schema_version": 1,
        "frozen_pack_sha256": sha256_path(frozen_pack_path),
        "replay_verdict": replay_verdict,
        "case_verdicts": [
            {
                "id": result.case_id,
                "expected_verdict": result.expected_verdict,
                "actual_verdict": result.actual_verdict,
            }
            for result in results
        ],
        "ledger_audit": [
            {
                "family": row.family,
                "classification": row.classification,
            }
            for row in audit_rows
        ],
    }
    _write_json(actual_snapshot_path, snapshot)
    golden_snapshot, golden_error = safe_load_structured_file(golden_snapshot_path)
    snapshot_matches = golden_error is None and golden_snapshot == snapshot
    mismatch_count = 0 if snapshot_matches else 1
    snapshot_report_path.write_text(
        _render_snapshot_report(golden_snapshot_path, actual_snapshot_path, snapshot_matches, mismatch_count),
        encoding="utf-8",
    )
    ci_report_path.write_text(
        _render_ci_report(Path(".github/workflows/claim-gate.yml"), {"verdict": replay_verdict, "snapshot_matches": snapshot_matches, "cases": replay_json["cases"]}),
        encoding="utf-8",
    )
    decision_memo_path.write_text(
        _render_decision_memo({"verdict": replay_verdict, "snapshot_matches": snapshot_matches, "cases": replay_json["cases"]}, audit_rows),
        encoding="utf-8",
    )

    return {
        "catalog_report": str(catalog_report_path),
        "replay_report": str(replay_report_path),
        "replay_csv": str(replay_csv_path),
        "replay_json": str(replay_json_path),
        "audit_report": str(audit_report_path),
        "discrepancy_report": str(discrepancy_report_path),
        "snapshot_report": str(snapshot_report_path),
        "ci_report": str(ci_report_path),
        "decision_memo": str(decision_memo_path),
        "snapshot_matches": snapshot_matches,
        "verdict": replay_verdict,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay the frozen DoorKey claim gate against the repo's historical result phases.")
    parser.add_argument("--frozen-pack", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--golden-snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_history_replay(
        Path(args.frozen_pack),
        Path(args.catalog),
        Path(args.golden_snapshot),
        Path(args.output_dir),
    )
    if result["verdict"] != "PASS: historical replay matches the expected claim-gate verdict map":
        raise SystemExit(result["verdict"])
    if not result["snapshot_matches"]:
        raise SystemExit("FAIL: historical replay snapshot drift detected")


if __name__ == "__main__":
    main()
