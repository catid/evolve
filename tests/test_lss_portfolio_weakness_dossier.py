from psmn_rl.analysis.lss_portfolio_weakness_dossier import _format_optional
from psmn_rl.analysis.lss_portfolio_signal_atlas import _classify_seed


def test_classify_seed_weakness() -> None:
    assert _classify_seed(0.984375, 1.0, 1.0) == "control_advantaged"


def test_classify_seed_support() -> None:
    assert _classify_seed(1.0, 0.5, 1.0) == "benchmark_support"


def test_format_optional_none() -> None:
    assert _format_optional(None) == "n/a"


def test_format_optional_float() -> None:
    assert _format_optional(1.23456) == "1.2346"
