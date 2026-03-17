from psmn_rl.analysis.lss_frozen_claim import _build_decision_memo, _build_final_block_single_expert_report, _build_updated_combined_report


def _row(lane: str, seed: int, label: str, value: float) -> dict[str, object]:
    return {
        "lane": lane,
        "seed": seed,
        "label": label,
        "variant": label,
        "method": label,
        "mode": "greedy",
        "eval_success_rate": value,
    }


def test_final_block_single_expert_report_calls_out_single_expert_closing_gap() -> None:
    rows = [
        _row("fresh_final", 47, "recovered_token_dense", 1.0),
        _row("fresh_final", 47, "kl_lss_token_dense", 1.0),
        _row("fresh_final", 47, "kl_lss_single_expert", 1.0),
        _row("fresh_final", 47, "baseline_sare", 0.0),
        _row("fresh_final", 47, "kl_lss_sare", 0.0),
        _row("fresh_final", 53, "recovered_token_dense", 1.0),
        _row("fresh_final", 53, "kl_lss_token_dense", 1.0),
        _row("fresh_final", 53, "kl_lss_single_expert", 0.8),
        _row("fresh_final", 53, "baseline_sare", 0.0),
        _row("fresh_final", 53, "kl_lss_sare", 0.5),
        _row("fresh_final", 59, "recovered_token_dense", 1.0),
        _row("fresh_final", 59, "kl_lss_token_dense", 1.0),
        _row("fresh_final", 59, "kl_lss_single_expert", 0.7),
        _row("fresh_final", 59, "baseline_sare", 0.0),
        _row("fresh_final", 59, "kl_lss_sare", 0.4),
    ]
    report = _build_final_block_single_expert_report(rows, episodes=64)
    assert "matched structured controls catch or beat" in report


def test_updated_combined_report_keeps_claim_frozen_when_single_expert_is_close() -> None:
    rows = [
        _row("original", 7, "recovered_token_dense", 0.7),
        _row("original", 7, "kl_lss_token_dense", 0.5),
        _row("original", 7, "kl_lss_single_expert", 1.0),
        _row("original", 7, "baseline_sare", 0.0),
        _row("original", 7, "kl_lss_sare", 0.75),
        _row("fresh_final", 47, "recovered_token_dense", 1.0),
        _row("fresh_final", 47, "kl_lss_token_dense", 0.5),
        _row("fresh_final", 47, "kl_lss_single_expert", 1.0),
        _row("fresh_final", 47, "baseline_sare", 0.0),
        _row("fresh_final", 47, "kl_lss_sare", 0.75),
    ]
    report = _build_updated_combined_report(rows, episodes=64)
    assert "not clearly enough over single_expert to thaw the claim" in report


def test_decision_memo_can_narrow_further() -> None:
    single_rows = [
        _row("fresh_final", 47, "kl_lss_token_dense", 1.0),
        _row("fresh_final", 47, "kl_lss_single_expert", 1.0),
        _row("fresh_final", 47, "kl_lss_sare", 0.0),
        _row("fresh_final", 53, "kl_lss_token_dense", 1.0),
        _row("fresh_final", 53, "kl_lss_single_expert", 0.8),
        _row("fresh_final", 53, "kl_lss_sare", 0.5),
    ]
    combined_rows = [
        _row("original", 7, "kl_lss_token_dense", 1.0),
        _row("original", 7, "kl_lss_single_expert", 1.0),
        _row("original", 7, "kl_lss_sare", 1.0),
        _row("fresh_final", 47, "kl_lss_token_dense", 1.0),
        _row("fresh_final", 47, "kl_lss_single_expert", 1.0),
        _row("fresh_final", 47, "kl_lss_sare", 0.0),
    ]
    memo = _build_decision_memo(single_rows, combined_rows, failure_rows=[])
    assert "narrower method-first result" in memo
