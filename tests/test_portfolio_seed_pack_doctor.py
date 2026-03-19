from copy import deepcopy
from pathlib import Path

from psmn_rl.analysis.lss_post_pass_campaign import _read_json
from psmn_rl.analysis.portfolio_seed_pack_doctor import evaluate_seed_pack_state


def test_evaluate_seed_pack_state_pass() -> None:
    validation = _read_json(Path("outputs/reports/portfolio_seed_pack_validation.json"))
    scorer = _read_json(Path("outputs/reports/portfolio_seed_pack_scorer.json"))
    result = evaluate_seed_pack_state(validation=validation, scorer=scorer)
    assert result["overall"] == "pass"


def test_evaluate_seed_pack_state_fail() -> None:
    validation = _read_json(Path("outputs/reports/portfolio_seed_pack_validation.json"))
    scorer = deepcopy(_read_json(Path("outputs/reports/portfolio_seed_pack_scorer.json")))
    scorer["grouped"]["advance_for_broader_dev"] = ["round10"]
    for row in scorer["rows"]:
        if row["candidate"] == "round7":
            row["verdict"] = "hold_seed_clean_but_below_incumbent"
    result = evaluate_seed_pack_state(validation=validation, scorer=scorer)
    assert result["overall"] == "fail"
