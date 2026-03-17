from psmn_rl.analysis.lss_resume_gate import _build_decision_memo


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


def test_decision_memo_stays_frozen_when_no_actionable_mechanism() -> None:
    reproduced_rows = [
        _row("fresh_final", 47, "kl_lss_token_dense", 1.0),
        _row("fresh_final", 47, "kl_lss_single_expert", 0.45),
        _row("fresh_final", 47, "kl_lss_sare", 0.0),
        _row("fresh_final", 53, "kl_lss_token_dense", 1.0),
        _row("fresh_final", 53, "kl_lss_single_expert", 0.52),
        _row("fresh_final", 53, "kl_lss_sare", 0.52),
        _row("fresh_final", 59, "kl_lss_token_dense", 1.0),
        _row("fresh_final", 59, "kl_lss_single_expert", 0.42),
        _row("fresh_final", 59, "kl_lss_sare", 0.42),
    ]
    failure_rows = [{"kind": "verdict", "resume_gate_status": "no actionable mechanism"}]
    combined_rows = [
        _row("original", 7, "kl_lss_token_dense", 1.0),
        _row("original", 7, "kl_lss_single_expert", 1.0),
        _row("original", 7, "kl_lss_sare", 1.0),
        _row("fresh_final", 47, "kl_lss_token_dense", 1.0),
        _row("fresh_final", 47, "kl_lss_single_expert", 0.45),
        _row("fresh_final", 47, "kl_lss_sare", 0.0),
    ]
    memo = _build_decision_memo(reproduced_rows, failure_rows, combined_rows)
    assert "stay frozen as-is" in memo
    assert "resume attempt justified: `no`" in memo
