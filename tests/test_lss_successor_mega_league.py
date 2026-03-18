from psmn_rl.analysis.lss_successor_mega_league import _decision_status, _verification_pass


def test_verification_pass_requires_consistent_reruns() -> None:
    campaign = {"selection": {"verification_tolerance": 0.05, "verification_margin_floor": -0.02}}
    original = {
        "candidate_mean": 0.84,
        "round6_mean": 0.83,
        "delta_vs_round6": 0.01,
        "candidate_failures": 0.0,
    }
    rerun_a = {"candidate_mean": 0.83, "candidate_failures": 0.0}
    rerun_b = {"candidate_mean": 0.82, "candidate_failures": 0.0}
    assert _verification_pass(campaign, original, rerun_a, rerun_b, 0.0)

    rerun_b["candidate_mean"] = 0.74
    assert not _verification_pass(campaign, original, rerun_a, rerun_b, 0.0)


def test_decision_status_confirms_round6_when_no_viable_challenger_and_gate_passes() -> None:
    campaign = {
        "current_canonical_name": "round6",
        "selection": {"pack_gate_required_verdict": "PASS: thaw consideration allowed"},
    }
    holdout = {"best_candidate": None}
    anti_regression = {"challenger_pass": False}
    route = {"round6_pass": True, "challenger_pass": False}
    stability = {"round6_pass": True, "challenger_pass": False}
    gate_payload = {"verdict": "PASS: thaw consideration allowed"}
    assert _decision_status(campaign, holdout, anti_regression, route, stability, gate_payload) == "round6 confirmed and sealed as active canonical DoorKey benchmark"


def test_decision_status_promotes_challenger_only_when_full_path_clears() -> None:
    campaign = {
        "current_canonical_name": "round6",
        "selection": {"pack_gate_required_verdict": "PASS: thaw consideration allowed"},
    }
    holdout = {"best_candidate": "round7"}
    anti_regression = {"challenger_pass": True}
    route = {"round6_pass": True, "challenger_pass": True}
    stability = {"round6_pass": True, "challenger_pass": True}
    gate_payload = {"verdict": "PASS: thaw consideration allowed"}
    assert _decision_status(campaign, holdout, anti_regression, route, stability, gate_payload) == "challenger replaces round6 as active canonical DoorKey benchmark"

    gate_payload = {"verdict": "FAIL: claim remains frozen"}
    assert _decision_status(campaign, holdout, anti_regression, route, stability, gate_payload) == "challenger fails and round6 remains active incumbent"
