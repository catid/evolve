from psmn_rl.analysis.portfolio_frontier_contract_loader import load_frontier_contract
from psmn_rl.analysis.portfolio_frontier_docs_audit import (
    FRONTIER_GUARD_WORKFLOW_CONTRACT_REPORT,
    FileExpectation,
    audit_expected_file,
    build_expectations,
)


def test_build_expectations_uses_contract_values() -> None:
    contract = load_frontier_contract()
    expectations = {row.path: row for row in build_expectations(contract)}
    readme = expectations["README.md"]
    assert contract.benchmark.active_candidate_pack in readme.required_snippets
    assert contract.frontier_roles.default_restart_prior in readme.required_snippets
    assert contract.frontier_roles.replay_validated_alternate in readme.required_snippets
    assert FRONTIER_GUARD_WORKFLOW_CONTRACT_REPORT in readme.required_snippets


def test_audit_expected_file_pass() -> None:
    expectation = FileExpectation(
        path="README.md",
        required_snippets=("round6", "round7", "round10"),
    )
    result = audit_expected_file(expectation, "round6 round7 round10")
    assert result["status"] == "pass"
    assert result["missing"] == []


def test_audit_expected_file_fail() -> None:
    expectation = FileExpectation(
        path="summary.md",
        required_snippets=("round6", "round7"),
    )
    result = audit_expected_file(expectation, "round6 only")
    assert result["status"] == "fail"
    assert result["missing"] == ["round7"]
