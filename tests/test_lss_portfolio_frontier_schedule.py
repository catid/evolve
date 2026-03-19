from psmn_rl.analysis.lss_portfolio_frontier_schedule import _priority


def test_priority_ordering() -> None:
    assert _priority("default_restart") < _priority("validated_alternate")
    assert _priority("validated_alternate") < _priority("hold_only")
    assert _priority("hold_only") < _priority("retired")
