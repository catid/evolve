from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CANDIDATE_PACK_PATH = Path("outputs/reports/portfolio_candidate_pack.json")


@dataclass(frozen=True)
class PackReference:
    path: str
    sha256: str


@dataclass(frozen=True)
class CandidatePackEvaluation:
    episodes: int
    path_key: str
    task: str


@dataclass(frozen=True)
class CandidatePackActiveState:
    active_pack_role: str
    winner: str
    challenger_viable_pre_gate: bool
    current_active_pack: PackReference
    archived_legacy_frozen_pack: PackReference


@dataclass(frozen=True)
class CandidatePackPortfolioCampaign:
    winner: str
    active_canonical_pack: str
    archived_legacy_pack: str
    gate_reference_pack: str


@dataclass(frozen=True)
class CandidatePackArtifact:
    path: str
    role: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class CandidatePack:
    candidate_name: str
    pack_type: str
    task: str
    requested_claims: tuple[str, ...]
    controls_present: tuple[str, ...]
    evaluation: CandidatePackEvaluation
    active_benchmark_state: CandidatePackActiveState
    frozen_pack_reference: PackReference
    portfolio_campaign: CandidatePackPortfolioCampaign
    artifacts: tuple[CandidatePackArtifact, ...]

    def artifact_by_role(self, role: str) -> CandidatePackArtifact:
        for artifact in self.artifacts:
            if artifact.role == role:
                return artifact
        raise KeyError(role)


def load_portfolio_candidate_pack(path: Path | None = None) -> CandidatePack:
    pack_path = path or DEFAULT_CANDIDATE_PACK_PATH
    data = json.loads(pack_path.read_text(encoding="utf-8"))
    return CandidatePack(
        candidate_name=str(data["candidate_name"]),
        pack_type=str(data["pack_type"]),
        task=str(data["task"]),
        requested_claims=tuple(str(value) for value in data["requested_claims"]),
        controls_present=tuple(str(value) for value in data["controls_present"]),
        evaluation=CandidatePackEvaluation(
            episodes=int(data["evaluation"]["episodes"]),
            path_key=str(data["evaluation"]["path_key"]),
            task=str(data["evaluation"]["task"]),
        ),
        active_benchmark_state=CandidatePackActiveState(
            active_pack_role=str(data["active_benchmark_state"]["active_pack_role"]),
            winner=str(data["active_benchmark_state"]["winner"]),
            challenger_viable_pre_gate=bool(data["active_benchmark_state"]["challenger_viable_pre_gate"]),
            current_active_pack=PackReference(
                path=str(data["active_benchmark_state"]["current_active_pack"]["path"]),
                sha256=str(data["active_benchmark_state"]["current_active_pack"]["sha256"]),
            ),
            archived_legacy_frozen_pack=PackReference(
                path=str(data["active_benchmark_state"]["archived_legacy_frozen_pack"]["path"]),
                sha256=str(data["active_benchmark_state"]["archived_legacy_frozen_pack"]["sha256"]),
            ),
        ),
        frozen_pack_reference=PackReference(
            path=str(data["frozen_pack_reference"]["path"]),
            sha256=str(data["frozen_pack_reference"]["sha256"]),
        ),
        portfolio_campaign=CandidatePackPortfolioCampaign(
            winner=str(data["portfolio_campaign"]["winner"]),
            active_canonical_pack=str(data["portfolio_campaign"]["future_comparison_policy"]["active_canonical_pack"]),
            archived_legacy_pack=str(data["portfolio_campaign"]["future_comparison_policy"]["archived_legacy_pack"]),
            gate_reference_pack=str(data["portfolio_campaign"]["future_comparison_policy"]["gate_reference_pack"]),
        ),
        artifacts=tuple(
            CandidatePackArtifact(
                path=str(row["path"]),
                role=str(row["role"]),
                sha256=str(row["sha256"]),
                size_bytes=int(row["size_bytes"]),
            )
            for row in data["artifacts"]
        ),
    )
