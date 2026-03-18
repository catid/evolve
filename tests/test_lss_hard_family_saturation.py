from __future__ import annotations

import json
from pathlib import Path

from psmn_rl.analysis.lss_hard_family_saturation import _block_specs, _stage2_pass, _stage3_pass, _stability_class


def test_stage2_pass_requires_dev_gain_gap_narrowing_and_no_new_failures() -> None:
    good = {
        "candidate_mean": 0.90,
        "incumbent_mean": 0.82,
        "gap_narrowing_vs_token": 0.08,
        "candidate_failures": 0,
        "max_block_delta": 0.06,
    }
    weak_gap = dict(good, gap_narrowing_vs_token=0.02)
    weak_block = dict(good, max_block_delta=0.03)
    weak_gain = dict(good, candidate_mean=0.83)
    assert _stage2_pass(good, incumbent_failures=0) is True
    assert _stage2_pass(weak_gap, incumbent_failures=0) is False
    assert _stage2_pass(weak_block, incumbent_failures=0) is False
    assert _stage2_pass(weak_gain, incumbent_failures=0) is False


def test_stage3_pass_uses_dev_split_not_per_block_token_tie() -> None:
    block_summaries = [
        {"lane": "post_pass_b", "candidate_minus_token": -0.10, "candidate_minus_single": 0.10},
        {"lane": "post_pass_c", "candidate_minus_token": 0.20, "candidate_minus_single": 0.05},
    ]
    family = {"candidate_minus_token": 0.02, "candidate_minus_single": 0.01, "candidate_failures": 0}
    assert _stage3_pass(block_summaries, family) is True
    assert _stage3_pass(block_summaries, {**family, "candidate_minus_token": -0.01}) is False
    assert _stage3_pass(block_summaries, {**family, "candidate_minus_single": -0.03}) is False


def test_stability_class_distinguishes_plateau_from_spike() -> None:
    assert _stability_class([0.40, 0.45, 0.44, 0.46]) == "stable_plateau"
    assert _stability_class([0.00, 0.05, 0.60, 0.10]) == "narrow_spike"
    assert _stability_class([0.10, 0.45, 0.30, 0.22]) == "noisy_brittle"


def test_block_specs_prefers_dynamic_definition_json(tmp_path: Path) -> None:
    definition_json = tmp_path / "definition.json"
    definition_json.write_text(
        json.dumps(
            {
                "hard_family_dev_blocks": [{"lane": "post_pass_b", "seeds": [73, 79, 83]}],
                "hard_family_holdout_blocks": [{"lane": "fresh_final", "seeds": [47, 53, 59]}],
            }
        ),
        encoding="utf-8",
    )
    campaign = {
        "reports": {"definition_json": str(definition_json)},
        "seed_groups": {
            "hard_family_dev": {"blocks": [{"lane": "fallback_dev", "seeds": [1]}]},
            "hard_family_holdout": {"blocks": [{"lane": "fallback_holdout", "seeds": [2]}]},
        },
    }
    assert _block_specs(campaign, "hard_family_dev") == [{"lane": "post_pass_b", "seeds": [73, 79, 83]}]
    assert _block_specs(campaign, "hard_family_holdout") == [{"lane": "fresh_final", "seeds": [47, 53, 59]}]
