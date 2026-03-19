from psmn_rl.analysis.lss_portfolio_round_prior_dossier import _bucket


def test_bucket_recommended_default() -> None:
    assert _bucket(0.0, 79000.0, 79841.8889) == "recommended_conservative_default"


def test_bucket_cheaper_but_below_incumbent() -> None:
    assert _bucket(-0.0521, 74232.0, 79841.8889) == "cheapest_but_below_incumbent"


def test_bucket_higher_cost_same_signal() -> None:
    assert _bucket(0.0, 86907.2, 79841.8889) == "higher_cost_same_signal"
