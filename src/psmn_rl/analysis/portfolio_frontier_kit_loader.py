from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_KIT_PATH = Path("outputs/reports/portfolio_frontier_kit.json")


@dataclass(frozen=True)
class KitBenchmark:
    active_candidate: str
    active_candidate_pack: str
    archived_frozen_pack: str


@dataclass(frozen=True)
class KitFrontierRoles:
    default_restart_prior: str
    replay_validated_alternate: str
    hold_only_priors: tuple[str, ...]
    retired_priors: tuple[str, ...]


@dataclass(frozen=True)
class KitNextRestart:
    primary: str
    secondary: str


@dataclass(frozen=True)
class KitQueueRow:
    action: str
    bucket: str
    candidate: str
    priority: int
    reason: str


@dataclass(frozen=True)
class KitPromotionRules:
    broader_dev_only_after_seed_clear: bool
    hold_on_weakness_tie: bool
    prune_on_guardrail_regression: bool
    prune_on_support_regression: bool


@dataclass(frozen=True)
class KitSeedContractEntry:
    lane: str
    seed: int
    required_min_success: float | None = None
    required_strictly_above: float | None = None
    mode: str | None = None


@dataclass(frozen=True)
class FrontierKit:
    benchmark: KitBenchmark
    frontier_roles: KitFrontierRoles
    next_restart: KitNextRestart
    ordered_queue: tuple[KitQueueRow, ...]
    promotion_rules: KitPromotionRules
    seed_contract: dict[str, KitSeedContractEntry]

    def queue_row_by_candidate(self, candidate: str) -> KitQueueRow:
        for row in self.ordered_queue:
            if row.candidate == candidate:
                return row
        raise KeyError(candidate)


def load_frontier_kit(path: Path | None = None) -> FrontierKit:
    kit_path = path or DEFAULT_KIT_PATH
    data = json.loads(kit_path.read_text(encoding="utf-8"))
    return FrontierKit(
        benchmark=KitBenchmark(
            active_candidate=str(data["benchmark"]["active_candidate"]),
            active_candidate_pack=str(data["benchmark"]["active_candidate_pack"]),
            archived_frozen_pack=str(data["benchmark"]["archived_frozen_pack"]),
        ),
        frontier_roles=KitFrontierRoles(
            default_restart_prior=str(data["frontier_roles"]["default_restart_prior"]),
            replay_validated_alternate=str(data["frontier_roles"]["replay_validated_alternate"]),
            hold_only_priors=tuple(str(value) for value in data["frontier_roles"]["hold_only_priors"]),
            retired_priors=tuple(str(value) for value in data["frontier_roles"]["retired_priors"]),
        ),
        next_restart=KitNextRestart(
            primary=str(data["next_restart"]["primary"]),
            secondary=str(data["next_restart"]["secondary"]),
        ),
        ordered_queue=tuple(
            KitQueueRow(
                action=str(row["action"]),
                bucket=str(row["bucket"]),
                candidate=str(row["candidate"]),
                priority=int(row["priority"]),
                reason=str(row["reason"]),
            )
            for row in data["ordered_queue"]
        ),
        promotion_rules=KitPromotionRules(
            broader_dev_only_after_seed_clear=bool(data["promotion_rules"]["broader_dev_only_after_seed_clear"]),
            hold_on_weakness_tie=bool(data["promotion_rules"]["hold_on_weakness_tie"]),
            prune_on_guardrail_regression=bool(data["promotion_rules"]["prune_on_guardrail_regression"]),
            prune_on_support_regression=bool(data["promotion_rules"]["prune_on_support_regression"]),
        ),
        seed_contract={
            str(role): KitSeedContractEntry(
                lane=str(row["lane"]),
                seed=int(row["seed"]),
                required_min_success=(
                    None if row.get("required_min_success") is None else float(row["required_min_success"])
                ),
                required_strictly_above=(
                    None
                    if row.get("required_strictly_above") is None
                    else float(row["required_strictly_above"])
                ),
                mode=None if row.get("mode") is None else str(row["mode"]),
            )
            for role, row in data["seed_contract"].items()
        },
    )
