from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SEED_PACK_PATH = Path("outputs/reports/portfolio_seed_pack.json")


@dataclass(frozen=True)
class SeedPackBenchmark:
    active_candidate_name: str
    active_candidate_pack: str
    archived_frozen_pack: str


@dataclass(frozen=True)
class SeedPackFrontier:
    restart_default: tuple[str, ...]
    reserve_priors: tuple[str, ...]
    retired_priors: tuple[str, ...]
    support_status: str
    surviving_candidates: tuple[str, ...]


@dataclass(frozen=True)
class SeedPackGeneratedFrom:
    active_candidate_pack: str
    restart_policy: str
    screening_spec: str


@dataclass(frozen=True)
class SeedRole:
    lane: str
    seed: int
    required_min_success: float | None = None
    required_behavior: str | None = None


@dataclass(frozen=True)
class SeedPackScreeningRules:
    advance_rule: str
    hold_rules: tuple[str, ...]
    prune_rules: tuple[str, ...]
    seed_roles: dict[str, SeedRole]


@dataclass(frozen=True)
class SeedPackCandidateRow:
    candidate: str
    guardrail_277: float
    policy_bucket: str
    screen_rule: str
    support_233: float | None
    tier: str
    weakness_269: float


@dataclass(frozen=True)
class PortfolioSeedPack:
    benchmark: SeedPackBenchmark
    campaign_name: str
    pack_type: str
    frontier: SeedPackFrontier
    generated_from: SeedPackGeneratedFrom
    screening_rules: SeedPackScreeningRules
    candidate_rows: tuple[SeedPackCandidateRow, ...]

    def row_by_candidate(self, candidate: str) -> SeedPackCandidateRow:
        for row in self.candidate_rows:
            if row.candidate == candidate:
                return row
        raise KeyError(candidate)


def load_portfolio_seed_pack(path: Path | None = None) -> PortfolioSeedPack:
    pack_path = path or DEFAULT_SEED_PACK_PATH
    data = json.loads(pack_path.read_text(encoding="utf-8"))
    return PortfolioSeedPack(
        benchmark=SeedPackBenchmark(
            active_candidate_name=str(data["benchmark"]["active_candidate_name"]),
            active_candidate_pack=str(data["benchmark"]["active_candidate_pack"]),
            archived_frozen_pack=str(data["benchmark"]["archived_frozen_pack"]),
        ),
        campaign_name=str(data["campaign_name"]),
        pack_type=str(data["pack_type"]),
        frontier=SeedPackFrontier(
            restart_default=tuple(str(value) for value in data["frontier"]["restart_default"]),
            reserve_priors=tuple(str(value) for value in data["frontier"]["reserve_priors"]),
            retired_priors=tuple(str(value) for value in data["frontier"]["retired_priors"]),
            support_status=str(data["frontier"]["support_status"]),
            surviving_candidates=tuple(str(value) for value in data["frontier"]["surviving_candidates"]),
        ),
        generated_from=SeedPackGeneratedFrom(
            active_candidate_pack=str(data["generated_from"]["active_candidate_pack"]),
            restart_policy=str(data["generated_from"]["restart_policy"]),
            screening_spec=str(data["generated_from"]["screening_spec"]),
        ),
        screening_rules=SeedPackScreeningRules(
            advance_rule=str(data["screening_rules"]["advance_rule"]),
            hold_rules=tuple(str(value) for value in data["screening_rules"]["hold_rules"]),
            prune_rules=tuple(str(value) for value in data["screening_rules"]["prune_rules"]),
            seed_roles={
                str(role): SeedRole(
                    lane=str(entry["lane"]),
                    seed=int(entry["seed"]),
                    required_min_success=(
                        None
                        if entry.get("required_min_success") is None
                        else float(entry["required_min_success"])
                    ),
                    required_behavior=(
                        None if entry.get("required_behavior") is None else str(entry["required_behavior"])
                    ),
                )
                for role, entry in data["screening_rules"]["seed_roles"].items()
            },
        ),
        candidate_rows=tuple(
            SeedPackCandidateRow(
                candidate=str(row["candidate"]),
                guardrail_277=float(row["guardrail_277"]),
                policy_bucket=str(row["policy_bucket"]),
                screen_rule=str(row["screen_rule"]),
                support_233=None if row.get("support_233") is None else float(row["support_233"]),
                tier=str(row["tier"]),
                weakness_269=float(row["weakness_269"]),
            )
            for row in data["candidate_rows"]
        ),
    )
