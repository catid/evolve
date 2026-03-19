from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONTRACT_PATH = Path("outputs/reports/portfolio_frontier_contract.json")


@dataclass(frozen=True)
class SeedRole:
    lane: str
    seed: int


@dataclass(frozen=True)
class BenchmarkState:
    active_candidate: str
    active_candidate_pack: str
    archived_frozen_pack: str


@dataclass(frozen=True)
class FrontierRoles:
    default_restart_prior: str
    replay_validated_alternate: str
    hold_only_priors: tuple[str, ...]
    retired_priors: tuple[str, ...]


@dataclass(frozen=True)
class FrontierContract:
    benchmark: BenchmarkState
    frontier_roles: FrontierRoles
    seed_roles: dict[str, SeedRole]
    global_hard: tuple[str, ...]
    differentiator: tuple[str, ...]
    support_min_success: float
    weakness_min_success_exclusive: float
    guardrail_min_success: float
    support_status: str

    def role_seed(self, role: str) -> SeedRole:
        return self.seed_roles[role]


def load_frontier_contract(path: Path | None = None) -> FrontierContract:
    contract_path = path or DEFAULT_CONTRACT_PATH
    data = json.loads(contract_path.read_text())
    return FrontierContract(
        benchmark=BenchmarkState(
            active_candidate=str(data["benchmark"]["active_candidate"]),
            active_candidate_pack=str(data["benchmark"]["active_candidate_pack"]),
            archived_frozen_pack=str(data["benchmark"]["archived_frozen_pack"]),
        ),
        frontier_roles=FrontierRoles(
            default_restart_prior=str(data["frontier_roles"]["default_restart_prior"]),
            replay_validated_alternate=str(data["frontier_roles"]["replay_validated_alternate"]),
            hold_only_priors=tuple(str(value) for value in data["frontier_roles"]["hold_only_priors"]),
            retired_priors=tuple(str(value) for value in data["frontier_roles"]["retired_priors"]),
        ),
        seed_roles={
            str(role): SeedRole(lane=str(row["lane"]), seed=int(row["seed"]))
            for role, row in data["seed_roles"].items()
        },
        global_hard=tuple(str(value) for value in data["seed_labels"]["global_hard"]),
        differentiator=tuple(str(value) for value in data["seed_labels"]["differentiator"]),
        support_min_success=float(data["screening_thresholds"]["support_min_success"]),
        weakness_min_success_exclusive=float(data["screening_thresholds"]["weakness_min_success_exclusive"]),
        guardrail_min_success=float(data["screening_thresholds"]["guardrail_min_success"]),
        support_status=str(data["frontier_support_status"]),
    )
