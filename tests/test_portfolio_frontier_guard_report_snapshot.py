from __future__ import annotations

import json
from pathlib import Path


REPORT_PATH = Path("outputs/reports/portfolio_frontier_guard_report.json")


def load_report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_guard_report_snapshot() -> None:
    report = load_report()
    assert report["overall"] == "pass"
    assert report["active_candidate"] == "round6"
    assert report["active_candidate_pack"] == "outputs/reports/portfolio_candidate_pack.json"
    assert report["archived_frozen_pack"] == "outputs/reports/frozen_benchmark_pack.json"
    assert report["default_restart_prior"] == "round7"
    assert report["replay_validated_alternate"] == "round10"
    assert [check["label"] for check in report["checks"]] == [
        "consistency_pass",
        "docs_audit_pass",
        "workflow_contract_pass",
        "doctor_pass",
        "seed_pack_doctor_pass",
        "active_state_doctor_pass",
    ]
    assert all(check["status"] == "pass" for check in report["checks"])
