from __future__ import annotations

from psmn_rl.analysis.lss_claim_consolidation import (
    _build_combined_doorkey_report,
    _build_fresh_matched_control_report,
)


def _row(lane: str, seed: int, label: str, success: float) -> dict[str, object]:
    return {
        "lane": lane,
        "seed": seed,
        "label": label,
        "mode": "greedy",
        "eval_success_rate": success,
    }


def test_fresh_matched_control_report_mentions_narrow_when_token_matches() -> None:
    rows = [
        _row("fresh", 23, "flat_dense", 1.0),
        _row("fresh", 23, "recovered_token_dense", 0.5),
        _row("fresh", 23, "kl_lss_token_dense", 1.0),
        _row("fresh", 23, "baseline_sare", 0.0),
        _row("fresh", 23, "kl_lss_sare", 1.0),
        _row("fresh", 29, "flat_dense", 1.0),
        _row("fresh", 29, "recovered_token_dense", 0.0),
        _row("fresh", 29, "kl_lss_token_dense", 1.0),
        _row("fresh", 29, "baseline_sare", 0.0),
        _row("fresh", 29, "kl_lss_sare", 1.0),
    ]
    report = _build_fresh_matched_control_report(rows, episodes=64)
    assert "competitive" in report or "harden" in report


def test_combined_doorkey_report_mentions_method_first_when_token_wins() -> None:
    rows = [
        _row("original", 7, "flat_dense", 1.0),
        _row("original", 7, "recovered_token_dense", 0.7),
        _row("original", 7, "kl_lss_token_dense", 0.9),
        _row("original", 7, "baseline_sare", 0.0),
        _row("original", 7, "kl_lss_sare", 0.8),
        _row("fresh", 23, "flat_dense", 1.0),
        _row("fresh", 23, "recovered_token_dense", 0.3),
        _row("fresh", 23, "kl_lss_token_dense", 1.0),
        _row("fresh", 23, "baseline_sare", 0.0),
        _row("fresh", 23, "kl_lss_sare", 0.7),
    ]
    report = _build_combined_doorkey_report(rows, episodes=64)
    assert "method-first" in report
