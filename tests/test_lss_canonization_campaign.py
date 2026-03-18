from __future__ import annotations

from psmn_rl.analysis.lss_canonization_campaign import _stage2_pass, _stage3_pass, _stability_class


def test_stage2_pass_requires_material_gain_gap_narrowing_and_no_new_failures() -> None:
    good = {
        "candidate_mean": 0.92,
        "incumbent_mean": 0.85,
        "gap_narrowing_vs_token": 0.08,
        "candidate_failures": 0,
        "post_pass_b_delta": 0.06,
    }
    weak_gap = dict(good, gap_narrowing_vs_token=0.03)
    weak_gain = dict(good, candidate_mean=0.86)
    assert _stage2_pass(good, incumbent_failures=0) is True
    assert _stage2_pass(weak_gap, incumbent_failures=0) is False
    assert _stage2_pass(weak_gain, incumbent_failures=0) is False


def test_stage3_pass_requires_fixing_post_pass_b_and_family_token_gap() -> None:
    block_summaries = [
        {"lane": "post_pass_b", "candidate_minus_token": 0.00, "candidate_minus_single": 0.05},
        {"lane": "post_pass_c", "candidate_minus_token": 0.10, "candidate_minus_single": 0.02},
    ]
    family = {"candidate_minus_token": 0.04, "candidate_minus_single": 0.01, "candidate_failures": 0}
    assert _stage3_pass(block_summaries, family) is True
    assert _stage3_pass([{**block_summaries[0], "candidate_minus_token": -0.01}, block_summaries[1]], family) is False
    assert _stage3_pass(block_summaries, {**family, "candidate_minus_token": -0.02}) is False


def test_stability_class_distinguishes_plateau_from_spike() -> None:
    assert _stability_class([0.40, 0.45, 0.44, 0.46]) == "stable_plateau"
    assert _stability_class([0.00, 0.05, 0.60, 0.10]) == "narrow_spike"
    assert _stability_class([0.10, 0.45, 0.30, 0.22]) == "noisy_brittle"
