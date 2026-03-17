from __future__ import annotations

import json
from pathlib import Path

from psmn_rl.analysis.claim_gate_conformance import run_conformance


def test_claim_gate_conformance_matches_golden_snapshot(tmp_path: Path) -> None:
    output_dir = tmp_path / "claim_gate_corpus"
    result = run_conformance(
        Path("outputs/reports/frozen_benchmark_pack.json"),
        Path("outputs/reports/frozen_candidate_result_pack.json"),
        Path("tests/data/claim_gate_corpus/corpus.yaml"),
        output_dir,
        Path("tests/data/claim_gate_corpus/golden_snapshot.json"),
    )
    assert result["conformance_verdict"] == "PASS: claim-gate conformance suite matched the expected corpus verdicts"
    assert result["snapshot_matches"] is True
    report = json.loads(Path(result["conformance_json"]).read_text(encoding="utf-8"))
    assert all(case["passed"] for case in report["cases"])
