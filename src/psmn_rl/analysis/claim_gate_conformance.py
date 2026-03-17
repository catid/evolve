from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from psmn_rl.analysis.benchmark_pack import load_structured_file, safe_load_structured_file, sha256_path, validate_frozen_benchmark_pack
from psmn_rl.analysis.claim_gate import evaluate_pack_claim_gate_from_paths, render_pack_gate_report


@dataclass(slots=True)
class CorpusCaseResult:
    case_id: str
    category: str
    description: str
    expected_verdict: str
    actual_verdict: str
    passed: bool
    rationale: str
    candidate_pack: str
    gate_report: str
    gate_json: str
    notable_checks: list[str]


HARDENING_FINDINGS = [
    {
        "weakness": "Malformed pack files could raise loader errors instead of producing a structured gate verdict.",
        "fix": "The pack-based gate now uses safe structured loads and returns `INCONCLUSIVE: missing prerequisites` with concrete `*_pack_load` reasons.",
        "cases": ["schema_malformed_json"],
    },
    {
        "weakness": "Disallowed claim widening could be downgraded to `INCONCLUSIVE` when the same pack also omitted controls or metrics.",
        "fix": "Claim-scope failures now dominate pack-validation incompleteness, so overclaim attempts stay hard-failed.",
        "cases": ["overclaim_keycorridor_transfer", "overclaim_missing_controls"],
    },
    {
        "weakness": "Candidate-pack type, metrics, actual-set, artifact, and provenance fields accepted overly loose shapes.",
        "fix": "The validator now enforces list/mapping/number/boolean field shapes and rejects malformed lane-seed payloads, duplicate artifact roles, and invalid git provenance fields.",
        "cases": [
            "schema_wrong_schema_version",
            "schema_wrong_controls_type",
            "schema_missing_actual_sets",
            "semantic_wrong_retry_seeds",
            "provenance_missing_file",
        ],
    },
    {
        "weakness": "A candidate pack could declare metrics that diverged from its `candidate_metrics_json` artifact.",
        "fix": "The validator now parses `candidate_metrics_json` and checks that task, evaluation, claims, controls, metrics, and actual sets match the candidate pack.",
        "cases": ["tampered_metrics_artifact_mismatch"],
    },
]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _split_path(path: str) -> list[str]:
    return [segment for segment in path.split(".") if segment]


def _walk_parent(payload: Any, path: str, *, create: bool) -> tuple[Any, str]:
    segments = _split_path(path)
    if not segments:
        raise ValueError("path must not be empty")
    current = payload
    for segment in segments[:-1]:
        if isinstance(current, dict):
            if segment not in current:
                if not create:
                    raise KeyError(path)
                current[segment] = {}
            current = current[segment]
            continue
        if isinstance(current, list):
            if not segment.isdigit():
                raise KeyError(path)
            index = int(segment)
            if index >= len(current):
                if not create:
                    raise KeyError(path)
                current.extend({} for _ in range(index - len(current) + 1))
            current = current[index]
            continue
        raise KeyError(path)
    return current, segments[-1]


def _set_path(payload: Any, path: str, value: Any) -> None:
    parent, leaf = _walk_parent(payload, path, create=True)
    if isinstance(parent, dict):
        parent[leaf] = value
        return
    if isinstance(parent, list) and leaf.isdigit():
        index = int(leaf)
        if index >= len(parent):
            parent.extend(None for _ in range(index - len(parent) + 1))
        parent[index] = value
        return
    raise KeyError(path)


def _delete_path(payload: Any, path: str) -> None:
    parent, leaf = _walk_parent(payload, path, create=False)
    if isinstance(parent, dict):
        parent.pop(leaf, None)
        return
    if isinstance(parent, list) and leaf.isdigit():
        index = int(leaf)
        if 0 <= index < len(parent):
            parent.pop(index)
        return
    raise KeyError(path)


def _sync_candidate_metrics_artifact(candidate: dict[str, Any], output_dir: Path, case_id: str) -> None:
    metrics_payload = {
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
    metrics_path = output_dir / "artifacts" / f"{case_id}_candidate_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")
    for artifact in candidate.get("artifacts", []):
        if artifact.get("role") != "candidate_metrics_json":
            continue
        artifact["path"] = str(metrics_path)
        artifact["sha256"] = sha256_path(metrics_path)
        artifact["size_bytes"] = metrics_path.stat().st_size
        artifact["exists"] = True
        break


def _materialize_candidate_case(
    spec: dict[str, Any],
    base_candidate: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, dict[str, Any] | None]:
    case_id = str(spec["id"])
    case_dir = output_dir / "packs"
    case_dir.mkdir(parents=True, exist_ok=True)

    source_file = spec.get("source_file")
    if source_file:
        source_path = Path(str(source_file))
        candidate_path = case_dir / f"{case_id}{source_path.suffix.lower() or '.json'}"
        candidate_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
        return candidate_path, None

    candidate = copy.deepcopy(base_candidate)
    candidate["candidate_name"] = case_id
    for path, value in spec.get("set", {}).items():
        _set_path(candidate, str(path), value)
    for path in spec.get("delete", []):
        _delete_path(candidate, str(path))
    if spec.get("sync_candidate_metrics_artifact", True):
        _sync_candidate_metrics_artifact(candidate, output_dir, case_id)
    for path, value in spec.get("post_set", {}).items():
        _set_path(candidate, str(path), value)
    for path in spec.get("post_delete", []):
        _delete_path(candidate, str(path))
    candidate_path = case_dir / f"{case_id}.json"
    candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True), encoding="utf-8")
    return candidate_path, candidate


def _notable_checks(checks: list[dict[str, str]]) -> list[str]:
    return [f"{check['name']}:{check['status']}" for check in checks if check["status"] != "PASS"][:6]


def _frozen_pack_summary(frozen_pack: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_id": frozen_pack.get("claim", {}).get("id"),
        "claim_status": frozen_pack.get("claim", {}).get("status"),
        "allowed_claim_key": frozen_pack.get("claim", {}).get("allowed_claim_key"),
        "disallowed_claim_keys": frozen_pack.get("claim", {}).get("disallowed_claim_keys", []),
        "evaluation": frozen_pack.get("evaluation", {}),
        "required_controls": frozen_pack.get("thaw_gate", {}).get("required_controls", []),
        "retry_block_thresholds": frozen_pack.get("thaw_gate", {}).get("retry_block", {}),
        "combined_thresholds": frozen_pack.get("thaw_gate", {}).get("combined_picture", {}),
        "candidate_pack_schema_version": frozen_pack.get("candidate_pack", {}).get("schema_version"),
    }


def _render_corpus_report(
    corpus_name: str,
    results: list[CorpusCaseResult],
) -> str:
    lines = [
        "# Claim Gate Corpus Report",
        "",
        f"- corpus: `{corpus_name}`",
        f"- case count: `{len(results)}`",
        "",
        "| Case | Category | Expected | Rationale | Candidate Pack |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| `{result.case_id}` | `{result.category}` | `{result.expected_verdict}` | {result.rationale} | `{result.candidate_pack}` |"
        )
    return "\n".join(lines) + "\n"


def _render_conformance_report(
    frozen_pack_path: Path,
    frozen_pack_verdict: str,
    results: list[CorpusCaseResult],
) -> str:
    passed = sum(1 for result in results if result.passed)
    lines = [
        "# Claim Gate Conformance Report",
        "",
        f"- frozen benchmark pack: `{frozen_pack_path}`",
        f"- frozen pack validation verdict: `{frozen_pack_verdict}`",
        f"- conformance cases passed: `{passed}/{len(results)}`",
        "",
        "| Case | Expected | Actual | Result | Notable Checks | Gate Report |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        checks = ", ".join(f"`{item}`" for item in result.notable_checks) or "`all-pass`"
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"| `{result.case_id}` | `{result.expected_verdict}` | `{result.actual_verdict}` | `{status}` | {checks} | `{result.gate_report}` |"
        )
    verdict = "PASS: claim-gate conformance suite matched the expected corpus verdicts" if passed == len(results) else "FAIL: claim-gate conformance mismatches detected"
    lines.extend(["", "## Verdict", "", verdict])
    return "\n".join(lines) + "\n"


def _render_hardening_report(results: list[CorpusCaseResult]) -> str:
    case_ids = {result.case_id for result in results}
    lines = [
        "# Claim Gate Hardening Report",
        "",
        "This report records the concrete weaknesses exercised by the adversarial corpus and the hardening added to block them.",
        "",
        "| Weakness | Fix | Covered Cases | Coverage |",
        "| --- | --- | --- | --- |",
    ]
    for finding in HARDENING_FINDINGS:
        covered_cases = [case_id for case_id in finding["cases"] if case_id in case_ids]
        coverage = "PASS" if covered_cases else "FAIL"
        lines.append(
            f"| {finding['weakness']} | {finding['fix']} | {', '.join(f'`{case}`' for case in covered_cases) or '-'} | `{coverage}` |"
        )
    lines.extend(
        [
            "",
            "## Outcome",
            "",
            "- Malformed candidate files are now converted into structured `INCONCLUSIVE` gate results.",
            "- Disallowed claim widening stays `FAIL` even when the same candidate also omits controls or metrics.",
            "- Candidate packs now fail validation on malformed field shapes, bad provenance, and metrics-artifact mismatches.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_golden_snapshot_report(
    golden_path: Path,
    actual_snapshot_path: Path,
    snapshot_matches: bool,
    mismatch_count: int,
) -> str:
    verdict = "PASS: golden snapshot matches current frozen pack and gate corpus" if snapshot_matches else "FAIL: golden snapshot drift detected"
    lines = [
        "# Golden Snapshot Report",
        "",
        f"- golden snapshot: `{golden_path}`",
        f"- actual snapshot: `{actual_snapshot_path}`",
        f"- mismatches: `{mismatch_count}`",
        "",
        "## Verdict",
        "",
        verdict,
    ]
    return "\n".join(lines) + "\n"


def _render_ci_report(conformance_json: dict[str, Any], workflow_path: Path) -> str:
    lines = [
        "# CI Claim Gate Conformance Report",
        "",
        f"- workflow: `{workflow_path}`",
        f"- conformance verdict: `{conformance_json['verdict']}`",
        f"- covered corpus cases: `{len(conformance_json['cases'])}`",
        "",
        "## Workflow Coverage",
        "",
        "- validates the frozen benchmark pack",
        "- runs the adversarial claim-gate conformance corpus",
        "- runs the existing pack-based dry run",
        "- runs the full test suite",
        "",
        "The workflow now checks more than the happy path and will fail on claim-gate verdict drift.",
    ]
    return "\n".join(lines) + "\n"


def _render_decision_memo(conformance_json: dict[str, Any], snapshot_matches: bool) -> str:
    verdict = conformance_json["verdict"]
    lines = [
        "# Claim Gate Red-Team Decision Memo",
        "",
        "## Coverage",
        "",
        f"- adversarial corpus cases: `{len(conformance_json['cases'])}`",
        "- categories covered: happy-path reference, synthetic pass reference, incomplete packs, schema-invalid packs, semantically invalid packs, near-miss packs, provenance tampering packs, and overclaim packs",
        "",
        "## Findings",
        "",
        "- The gate now distinguishes `PASS`, `FAIL`, and `INCONCLUSIVE` across the adversarial corpus without relying on manual report inspection.",
        "- Disallowed claim widening is now hard-failed even when paired with missing controls or malformed fields.",
        "- Candidate pack tampering against `candidate_metrics_json` is now blocked by explicit consistency checks.",
        "- Malformed structured inputs now yield concrete `INCONCLUSIVE` reasons instead of uncaught loader errors.",
        "",
        "## Snapshot and Workflow",
        "",
        f"- golden snapshot status: `{'PASS' if snapshot_matches else 'FAIL'}`",
        "- CI now runs the conformance suite in addition to the frozen-pack validation and full tests.",
        "",
        "## Final Result",
        "",
        f"- conformance suite verdict: `{verdict}`",
        "- operational state: frozen DoorKey claim remains sealed behind the hardened pack-based gate",
        "- thaw triage surface: candidate result pack + pack-based claim gate only",
    ]
    return "\n".join(lines) + "\n"


def run_conformance(
    frozen_pack_path: Path,
    base_candidate_path: Path,
    corpus_path: Path,
    output_dir: Path,
    golden_snapshot_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_pack = load_structured_file(frozen_pack_path)
    frozen_pack_verdict, frozen_pack_checks = validate_frozen_benchmark_pack(frozen_pack)
    base_candidate = load_structured_file(base_candidate_path)
    corpus = _load_yaml(corpus_path)
    results: list[CorpusCaseResult] = []

    for case in corpus.get("cases", []):
        candidate_path, _candidate = _materialize_candidate_case(case, base_candidate, output_dir)
        verdict, checks, frozen_rows, candidate_rows = evaluate_pack_claim_gate_from_paths(frozen_pack_path, candidate_path)
        gate_report_path = output_dir / "gate_reports" / f"{case['id']}.md"
        gate_json_path = output_dir / "gate_reports" / f"{case['id']}.json"
        gate_report_path.parent.mkdir(parents=True, exist_ok=True)
        gate_report_path.write_text(
            render_pack_gate_report(
                frozen_pack_path,
                candidate_path,
                verdict,
                frozen_rows,
                candidate_rows,
                checks,
            ),
            encoding="utf-8",
        )
        gate_json_path.write_text(
            json.dumps(
                {
                    "frozen_pack": str(frozen_pack_path),
                    "candidate_pack": str(candidate_path),
                    "verdict": verdict,
                    "frozen_pack_validation": frozen_rows,
                    "candidate_pack_validation": candidate_rows,
                    "checks": [asdict(check) for check in checks],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        results.append(
            CorpusCaseResult(
                case_id=str(case["id"]),
                category=str(case["category"]),
                description=str(case["description"]),
                expected_verdict=str(case["expected_verdict"]),
                actual_verdict=verdict,
                passed=verdict == str(case["expected_verdict"]),
                rationale=str(case["rationale"]),
                candidate_pack=str(candidate_path),
                gate_report=str(gate_report_path),
                gate_json=str(gate_json_path),
                notable_checks=_notable_checks([asdict(check) for check in checks]),
            )
        )

    snapshot = {
        "schema_version": 1,
        "frozen_pack_summary": _frozen_pack_summary(frozen_pack),
        "frozen_candidate_verdict": next(
            result.actual_verdict for result in results if result.case_id == "happy_path_reference_fail"
        ),
        "case_verdicts": [
            {
                "id": result.case_id,
                "expected_verdict": result.expected_verdict,
                "actual_verdict": result.actual_verdict,
            }
            for result in results
        ],
    }
    actual_snapshot_path = output_dir / "claim_gate_golden_snapshot_actual.json"
    actual_snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")

    golden_snapshot, golden_error = safe_load_structured_file(golden_snapshot_path)
    snapshot_matches = golden_error is None and golden_snapshot == snapshot
    mismatch_count = 0 if snapshot_matches else 1

    corpus_report_path = output_dir.parent / "claim_gate_corpus_report.md"
    conformance_report_path = output_dir.parent / "claim_gate_conformance_report.md"
    conformance_json_path = output_dir.parent / "claim_gate_conformance_report.json"
    hardening_report_path = output_dir.parent / "claim_gate_hardening_report.md"
    golden_report_path = output_dir.parent / "golden_snapshot_report.md"
    ci_report_path = output_dir.parent / "ci_claim_gate_conformance_report.md"
    decision_memo_path = output_dir.parent / "claim_gate_redteam_decision_memo.md"

    conformance_verdict = (
        "PASS: claim-gate conformance suite matched the expected corpus verdicts"
        if all(result.passed for result in results)
        else "FAIL: claim-gate conformance mismatches detected"
    )
    conformance_json = {
        "frozen_pack": str(frozen_pack_path),
        "frozen_pack_validation_verdict": frozen_pack_verdict,
        "verdict": conformance_verdict,
        "golden_snapshot_matches": snapshot_matches,
        "cases": [asdict(result) for result in results],
    }

    corpus_report_path.write_text(_render_corpus_report(str(corpus.get("name", corpus_path)), results), encoding="utf-8")
    conformance_report_path.write_text(
        _render_conformance_report(frozen_pack_path, frozen_pack_verdict, results),
        encoding="utf-8",
    )
    conformance_json_path.write_text(json.dumps(conformance_json, indent=2, sort_keys=True), encoding="utf-8")
    hardening_report_path.write_text(_render_hardening_report(results), encoding="utf-8")
    golden_report_path.write_text(
        _render_golden_snapshot_report(golden_snapshot_path, actual_snapshot_path, snapshot_matches, mismatch_count),
        encoding="utf-8",
    )
    ci_report_path.write_text(
        _render_ci_report(conformance_json, Path(".github/workflows/claim-gate.yml")),
        encoding="utf-8",
    )
    decision_memo_path.write_text(
        _render_decision_memo(conformance_json, snapshot_matches),
        encoding="utf-8",
    )

    return {
        "corpus_report": str(corpus_report_path),
        "conformance_report": str(conformance_report_path),
        "conformance_json": str(conformance_json_path),
        "hardening_report": str(hardening_report_path),
        "golden_snapshot_report": str(golden_report_path),
        "ci_report": str(ci_report_path),
        "decision_memo": str(decision_memo_path),
        "snapshot_matches": snapshot_matches,
        "conformance_verdict": conformance_verdict,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the adversarial claim-gate conformance corpus.")
    parser.add_argument("--frozen-pack", required=True)
    parser.add_argument("--base-candidate", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--golden-snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_conformance(
        Path(args.frozen_pack),
        Path(args.base_candidate),
        Path(args.corpus),
        Path(args.output_dir),
        Path(args.golden_snapshot),
    )
    if result["conformance_verdict"] != "PASS: claim-gate conformance suite matched the expected corpus verdicts":
        raise SystemExit(result["conformance_verdict"])
    if not result["snapshot_matches"]:
        raise SystemExit("FAIL: golden snapshot drift detected")


if __name__ == "__main__":
    main()
