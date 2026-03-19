from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OPERATIONAL_STATE_PATH = Path("outputs/reports/portfolio_operational_state.json")


@dataclass(frozen=True)
class PortfolioOperationalState:
    status_type: str
    active_candidate: str
    active_candidate_pack: str
    archived_frozen_pack: str
    gate_verdict: str
    gate_mode: str
    active_pack_role: str
    default_restart_prior: str
    replay_validated_alternate: str
    evaluation_task: str
    evaluation_path_key: str
    evaluation_episodes: int
    frontier_guard_overall: str
    active_state_doctor_overall: str
    ready_for_bounded_restart: bool
    decision_reference: str
    gate_report_reference: str


def load_portfolio_operational_state(path: Path | None = None) -> PortfolioOperationalState:
    report_path = path or DEFAULT_OPERATIONAL_STATE_PATH
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return PortfolioOperationalState(
        status_type=str(data["status_type"]),
        active_candidate=str(data["active_candidate"]),
        active_candidate_pack=str(data["active_candidate_pack"]),
        archived_frozen_pack=str(data["archived_frozen_pack"]),
        gate_verdict=str(data["gate_verdict"]),
        gate_mode=str(data["gate_mode"]),
        active_pack_role=str(data["active_pack_role"]),
        default_restart_prior=str(data["default_restart_prior"]),
        replay_validated_alternate=str(data["replay_validated_alternate"]),
        evaluation_task=str(data["evaluation_task"]),
        evaluation_path_key=str(data["evaluation_path_key"]),
        evaluation_episodes=int(data["evaluation_episodes"]),
        frontier_guard_overall=str(data["frontier_guard_overall"]),
        active_state_doctor_overall=str(data["active_state_doctor_overall"]),
        ready_for_bounded_restart=bool(data["ready_for_bounded_restart"]),
        decision_reference=str(data["decision_reference"]),
        gate_report_reference=str(data["gate_report_reference"]),
    )
