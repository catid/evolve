from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_GUARD_REPORT_PATH = Path("outputs/reports/portfolio_frontier_guard_report.json")


@dataclass(frozen=True)
class GuardCheck:
    label: str
    status: str
    detail: str


@dataclass(frozen=True)
class FrontierGuardReport:
    overall: str
    active_candidate: str
    active_candidate_pack: str
    archived_frozen_pack: str
    default_restart_prior: str
    replay_validated_alternate: str
    checks: tuple[GuardCheck, ...]

    def check_by_label(self, label: str) -> GuardCheck:
        for check in self.checks:
            if check.label == label:
                return check
        raise KeyError(label)


def load_frontier_guard_report(path: Path | None = None) -> FrontierGuardReport:
    report_path = path or DEFAULT_GUARD_REPORT_PATH
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return FrontierGuardReport(
        overall=str(data["overall"]),
        active_candidate=str(data["active_candidate"]),
        active_candidate_pack=str(data["active_candidate_pack"]),
        archived_frozen_pack=str(data["archived_frozen_pack"]),
        default_restart_prior=str(data["default_restart_prior"]),
        replay_validated_alternate=str(data["replay_validated_alternate"]),
        checks=tuple(
            GuardCheck(
                label=str(row["label"]),
                status=str(row["status"]),
                detail=str(row["detail"]),
            )
            for row in data["checks"]
        ),
    )
