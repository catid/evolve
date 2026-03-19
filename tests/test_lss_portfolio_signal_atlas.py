from psmn_rl.analysis.lss_portfolio_signal_atlas import _classify_seed


def test_classify_seed_global_hard() -> None:
    assert _classify_seed(0.0, 0.0, 0.0) == "global_hard_failure"


def test_classify_seed_benchmark_support() -> None:
    assert _classify_seed(1.0, 0.625, 1.0) == "benchmark_support"


def test_classify_seed_control_advantaged() -> None:
    assert _classify_seed(0.75, 1.0, 0.5) == "control_advantaged"


def test_classify_seed_shared_parity() -> None:
    assert _classify_seed(1.0, 1.0, 1.0) == "shared_control_parity"
