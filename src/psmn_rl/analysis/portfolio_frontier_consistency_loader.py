from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONSISTENCY_REPORT_PATH = Path("outputs/reports/portfolio_frontier_consistency.json")


@dataclass(frozen=True)
class ConsistencyCheck:
    label: str
    status: str
    detail: str


@dataclass(frozen=True)
class FrontierConsistencyReport:
    overall: str
    checks: tuple[ConsistencyCheck, ...]

    def check_by_label(self, label: str) -> ConsistencyCheck:
        for check in self.checks:
            if check.label == label:
                return check
        raise KeyError(label)


def load_frontier_consistency_report(path: Path | None = None) -> FrontierConsistencyReport:
    report_path = path or DEFAULT_CONSISTENCY_REPORT_PATH
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return FrontierConsistencyReport(
        overall=str(data["overall"]),
        checks=tuple(
            ConsistencyCheck(
                label=str(row["label"]),
                status=str(row["status"]),
                detail=str(row["detail"]),
            )
            for row in data["checks"]
        ),
    )
