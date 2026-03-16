from __future__ import annotations

from psmn_rl.analysis.lss_claim_hardening import (
    _build_additional_seed_report,
    _build_matched_control_report,
    _build_transfer_report,
)


def _row(seed: int, label: str, success: float) -> dict[str, object]:
    return {
        "seed": seed,
        "label": label,
        "mode": "greedy",
        "eval_success_rate": success,
    }


def test_additional_seed_report_mentions_zero_failure() -> None:
    rows = [
        _row(23, "flat_dense", 1.0),
        _row(23, "recovered_token_dense", 0.5),
        _row(23, "baseline_sare", 0.0),
        _row(23, "kl_lss_sare", 0.0),
        _row(29, "flat_dense", 1.0),
        _row(29, "recovered_token_dense", 0.5),
        _row(29, "baseline_sare", 0.0),
        _row(29, "kl_lss_sare", 0.6),
    ]
    report = _build_additional_seed_report(rows, episodes=64)
    assert "complete greedy failure" in report


def test_matched_control_report_mentions_method_first_when_token_matches() -> None:
    rows = [
        _row(7, "flat_dense", 1.0),
        _row(7, "recovered_token_dense", 0.6),
        _row(7, "kl_lss_token_dense", 0.8),
        _row(7, "baseline_sare", 0.0),
        _row(7, "kl_lss_sare", 0.7),
        _row(11, "flat_dense", 1.0),
        _row(11, "recovered_token_dense", 0.4),
        _row(11, "kl_lss_token_dense", 0.8),
        _row(11, "baseline_sare", 0.0),
        _row(11, "kl_lss_sare", 0.7),
    ]
    report = _build_matched_control_report(rows, episodes=64)
    assert "method-first" in report


def test_transfer_report_mentions_no_transfer_for_zero_sare() -> None:
    rows = [
        _row(7, "flat_dense", 1.0),
        _row(7, "recovered_token_dense", 0.5),
        _row(7, "baseline_sare", 0.0),
        _row(7, "kl_lss_sare", 0.0),
        _row(11, "flat_dense", 1.0),
        _row(11, "recovered_token_dense", 0.4),
        _row(11, "baseline_sare", 0.0),
        _row(11, "kl_lss_sare", 0.0),
    ]
    report = _build_transfer_report(rows, episodes=64)
    assert "No transfer" in report
