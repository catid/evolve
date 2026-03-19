from psmn_rl.analysis.lss_portfolio_screening_spec import _score_rule


def test_score_rule_advance() -> None:
    assert _score_rule(1.0, 1.0, 1.0, 0.984375) == "advance_for_broader_dev"


def test_score_rule_prune_support_regression() -> None:
    assert _score_rule(0.4531, 1.0, 1.0, 0.984375) == "prune_support_regression"


def test_score_rule_prune_guardrail_regression() -> None:
    assert _score_rule(1.0, 1.0, 0.8906, 0.984375) == "prune_guardrail_regression"


def test_score_rule_tie_only() -> None:
    assert _score_rule(1.0, 0.984375, 1.0, 0.984375) == "tie_only_no_weakness_gain"


def test_score_rule_needs_support_measurement() -> None:
    assert _score_rule(None, 1.0, 1.0, 0.984375) == "needs_support_measurement"
