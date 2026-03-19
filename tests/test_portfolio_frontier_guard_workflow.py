from __future__ import annotations

from pathlib import Path

import yaml


WORKFLOW_PATH = Path(".github/workflows/portfolio-frontier-guard.yml")


def load_workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def workflow_on_block(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True, {}))


def test_pull_request_paths_cover_frontier_docs_audit_inputs() -> None:
    workflow = load_workflow()
    paths = workflow_on_block(workflow)["pull_request"]["paths"]
    required = {
        ".github/**",
        "README.md",
        "summary.md",
        "report.md",
        "scripts/run_portfolio_frontier_guard.sh",
        "src/psmn_rl/analysis/portfolio_candidate_pack_loader.py",
        "src/psmn_rl/analysis/portfolio_seed_pack_doctor.py",
        "src/psmn_rl/analysis/portfolio_seed_pack_doctor_loader.py",
        "src/psmn_rl/analysis/portfolio_seed_pack_loader.py",
        "src/psmn_rl/analysis/portfolio_seed_pack_scorer_loader.py",
        "src/psmn_rl/analysis/portfolio_seed_pack_validation_loader.py",
        "tests/test_portfolio_candidate_pack_loader.py",
        "tests/test_portfolio_seed_pack_doctor.py",
        "tests/test_portfolio_seed_pack_doctor_loader.py",
        "tests/test_portfolio_seed_pack_loader.py",
        "tests/test_portfolio_seed_pack_scorer_loader.py",
        "tests/test_portfolio_seed_pack_validation_loader.py",
        "outputs/reports/claim_ledger.md",
        "outputs/reports/portfolio_candidate_pack.json",
        "outputs/reports/portfolio_seed_pack*",
        "outputs/reports/portfolio_seed_pack.json",
        "outputs/reports/portfolio_gate_report.md",
        "outputs/reports/portfolio_decision_memo.md",
        "outputs/reports/portfolio_frontier_*",
    }
    assert required.issubset(set(paths))


def test_guard_job_runs_guard_script() -> None:
    workflow = load_workflow()
    steps = workflow["jobs"]["portfolio-frontier-guard"]["steps"]
    run_steps = [step["run"] for step in steps if "run" in step]
    assert any("bash -n scripts/run_portfolio_frontier_guard.sh" in run for run in run_steps)
    assert any("bash ./scripts/run_portfolio_frontier_guard.sh" in run for run in run_steps)
