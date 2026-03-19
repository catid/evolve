from __future__ import annotations

import json
from pathlib import Path


CONTRACT_PATH = Path("outputs/reports/portfolio_frontier_contract.json")


def _load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


def test_portfolio_frontier_contract_benchmark_state() -> None:
    contract = _load_contract()
    assert contract["benchmark"]["active_candidate"] == "round6"
    assert contract["benchmark"]["active_candidate_pack"] == "outputs/reports/portfolio_candidate_pack.json"
    assert contract["benchmark"]["archived_frozen_pack"] == "outputs/reports/frozen_benchmark_pack.json"


def test_portfolio_frontier_contract_roles() -> None:
    contract = _load_contract()
    assert contract["frontier_roles"]["default_restart_prior"] == "round7"
    assert contract["frontier_roles"]["replay_validated_alternate"] == "round10"
    assert contract["frontier_roles"]["hold_only_priors"] == ["round5"]
    assert contract["frontier_roles"]["retired_priors"] == ["door3_post5", "post_unlock_x5"]


def test_portfolio_frontier_contract_seed_roles() -> None:
    contract = _load_contract()
    assert contract["seed_roles"] == {
        "guardrail": {"lane": "prospective_h", "seed": 277},
        "ranking_support": {"lane": "prospective_f", "seed": 233},
        "ranking_weakness": {"lane": "prospective_h", "seed": 269},
        "sentinel": {"lane": "prospective_c", "seed": 193},
    }
    assert contract["seed_labels"] == {
        "differentiator": ["prospective_f:233"],
        "global_hard": ["prospective_c:193"],
    }
    assert contract["screening_thresholds"] == {
        "guardrail_min_success": 1.0,
        "support_min_success": 1.0,
        "weakness_min_success_exclusive": 0.984375001,
    }
