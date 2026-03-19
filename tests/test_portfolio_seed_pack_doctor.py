from dataclasses import replace

from psmn_rl.analysis.portfolio_seed_pack_doctor import evaluate_seed_pack_state
from psmn_rl.analysis.portfolio_seed_pack_scorer_loader import load_seed_pack_scorer_report
from psmn_rl.analysis.portfolio_seed_pack_validation_loader import load_seed_pack_validation_report


def test_evaluate_seed_pack_state_pass() -> None:
    validation = load_seed_pack_validation_report()
    scorer = load_seed_pack_scorer_report()
    result = evaluate_seed_pack_state(validation=validation, scorer=scorer)
    assert result["overall"] == "pass"


def test_evaluate_seed_pack_state_fail() -> None:
    validation = load_seed_pack_validation_report()
    scorer = load_seed_pack_scorer_report()
    scorer = replace(
        scorer,
        grouped={
            **scorer.grouped,
            "advance_for_broader_dev": ("round10",),
        },
        rows=tuple(
            replace(row, verdict="hold_seed_clean_but_below_incumbent")
            if row.candidate == "round7"
            else row
            for row in scorer.rows
        ),
    )
    result = evaluate_seed_pack_state(validation=validation, scorer=scorer)
    assert result["overall"] == "fail"
