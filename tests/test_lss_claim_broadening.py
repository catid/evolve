from __future__ import annotations

from psmn_rl.analysis.lss_claim_broadening import (
    _build_expanded_combined_report,
    _build_single_expert_report,
)


def _row(lane: str, seed: int, label: str, success: float) -> dict[str, object]:
    return {
        "lane": lane,
        "seed": seed,
        "label": label,
        "mode": "greedy",
        "eval_success_rate": success,
    }


def test_single_expert_report_mentions_sare_edge_when_it_stays_ahead() -> None:
    rows = [
        _row("original", 7, "flat_dense", 1.0),
        _row("original", 7, "recovered_token_dense", 0.6),
        _row("original", 7, "kl_lss_token_dense", 0.7),
        _row("original", 7, "baseline_sare", 0.0),
        _row("original", 7, "kl_lss_single_expert", 0.5),
        _row("original", 7, "kl_lss_sare", 1.0),
        _row("original", 11, "flat_dense", 1.0),
        _row("original", 11, "recovered_token_dense", 0.0),
        _row("original", 11, "kl_lss_token_dense", 0.0),
        _row("original", 11, "baseline_sare", 0.0),
        _row("original", 11, "kl_lss_single_expert", 0.2),
        _row("original", 11, "kl_lss_sare", 0.6),
    ]
    report = _build_single_expert_report(rows, episodes=64)
    assert "still stays ahead" in report


def test_expanded_combined_report_mentions_broadens_when_sare_leads() -> None:
    rows = [
        _row("original", 7, "flat_dense", 1.0),
        _row("original", 7, "recovered_token_dense", 0.7),
        _row("original", 7, "kl_lss_token_dense", 0.8),
        _row("original", 7, "baseline_sare", 0.0),
        _row("original", 7, "kl_lss_single_expert", 0.6),
        _row("original", 7, "kl_lss_sare", 1.0),
        _row("fresh", 23, "flat_dense", 1.0),
        _row("fresh", 23, "recovered_token_dense", 0.0),
        _row("fresh", 23, "kl_lss_token_dense", 0.5),
        _row("fresh", 23, "baseline_sare", 0.0),
        _row("fresh", 23, "kl_lss_sare", 1.0),
        _row("fresh_extra", 37, "flat_dense", 1.0),
        _row("fresh_extra", 37, "recovered_token_dense", 0.0),
        _row("fresh_extra", 37, "kl_lss_token_dense", 0.6),
        _row("fresh_extra", 37, "baseline_sare", 0.0),
        _row("fresh_extra", 37, "kl_lss_sare", 1.0),
    ]
    report = _build_expanded_combined_report(rows, episodes=64)
    assert "broadens the claim within scope" in report
