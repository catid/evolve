from psmn_rl.analysis.lss_successor_migration import _stage2_pass, _stage4_pass, _winner


def test_stage2_pass_requires_control_competition_and_meaningful_gain() -> None:
    campaign = {"selection": {"control_eps": 0.02, "margin_eps": 0.01}}
    round6 = {
        "sare_mean": 0.90,
        "token_mean": 0.88,
        "single_mean": 0.80,
        "sare_failures": 0.0,
        "sare_minus_token": 0.02,
        "sare_minus_single": 0.10,
    }
    challenger = {
        "sare_mean": 0.90,
        "token_mean": 0.89,
        "single_mean": 0.80,
        "sare_failures": 0.0,
        "sare_minus_token": 0.01,
        "sare_minus_single": 0.10,
    }
    assert not _stage2_pass(campaign, round6, challenger)

    challenger["sare_mean"] = 0.92
    challenger["sare_minus_token"] = 0.03
    assert _stage2_pass(campaign, round6, challenger)


def test_stage4_pass_rejects_healthy_regression() -> None:
    campaign = {"selection": {"anti_regression_tolerance": 0.02}}
    round6 = {"sare_mean": 0.95, "sare_failures": 0.0}
    challenger = {"sare_mean": 0.92, "sare_failures": 0.0}
    assert not _stage4_pass(campaign, round6, challenger)

    challenger["sare_mean"] = 0.94
    assert _stage4_pass(campaign, round6, challenger)


def test_winner_keeps_round6_when_challenger_fails_validation() -> None:
    campaign = {"current_canonical_name": "round6"}
    stage3 = {"best_candidate": "round7"}
    stage4 = {"challenger_pass": True}
    stage5 = {"challenger_pass": False}
    stage6 = {"challenger_pass": True}
    outcome = _winner(campaign, stage3, stage4, stage5, stage6)
    assert outcome["winner"] == "round6"
    assert not outcome["challenger_viable"]

    stage5["challenger_pass"] = True
    outcome = _winner(campaign, stage3, stage4, stage5, stage6)
    assert outcome["winner"] == "round7"
    assert outcome["challenger_viable"]
