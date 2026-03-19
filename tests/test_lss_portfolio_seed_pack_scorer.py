from psmn_rl.analysis.lss_portfolio_seed_pack_scorer import _score_candidate


def test_score_candidate_advances() -> None:
    assert _score_candidate(1.0, 1.0, 1.0, 0.984375, 0.0) == "advance_for_broader_dev"


def test_score_candidate_holds_below_incumbent() -> None:
    assert _score_candidate(1.0, 1.0, 1.0, 0.984375, -0.0521) == "hold_seed_clean_but_below_incumbent"


def test_score_candidate_prunes_support_regression() -> None:
    assert _score_candidate(0.4531, 1.0, 1.0, 0.984375, None) == "prune_support_regression"


def test_score_candidate_prunes_guardrail_regression() -> None:
    assert _score_candidate(None, 1.0, 0.8906, 0.984375, None) == "prune_guardrail_regression"


def test_score_candidate_holds_tie_only() -> None:
    assert _score_candidate(1.0, 0.984375, 1.0, 0.984375, 0.0) == "hold_tie_only"
