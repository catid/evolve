from psmn_rl.analysis.lss_portfolio_hard_seed_casebook import _classify_seed


def test_classify_seed_marks_global_hard_failure() -> None:
    rows = [
        {"sare": 0.0, "token_dense": 0.0, "single_expert": 0.0},
        {"sare": 0.0, "token_dense": 0.0, "single_expert": 0.0},
    ]
    assert _classify_seed(rows) == "global_hard_failure"


def test_classify_seed_marks_route_control_differentiator() -> None:
    rows = [
        {"sare": 1.0, "token_dense": 0.625, "single_expert": 1.0},
    ]
    assert _classify_seed(rows) == "route_control_differentiator"


def test_classify_seed_marks_shared_control_parity() -> None:
    rows = [
        {"sare": 1.0, "token_dense": 1.0, "single_expert": 1.0},
    ]
    assert _classify_seed(rows) == "shared_control_parity"
