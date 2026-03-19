from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SCHEDULE_PATH = Path("outputs/reports/portfolio_frontier_schedule.json")


@dataclass(frozen=True)
class ScheduleRow:
    action: str
    bucket: str
    candidate: str
    priority: int
    reason: str


@dataclass(frozen=True)
class SeedContractEntry:
    lane: str
    seed: int
    required_min_success: float | None = None
    required_strictly_above: float | None = None
    use: str | None = None


@dataclass(frozen=True)
class FrontierScheduleReport:
    active_benchmark: str
    rows: tuple[ScheduleRow, ...]
    seed_contract: dict[str, SeedContractEntry]

    def row_by_candidate(self, candidate: str) -> ScheduleRow:
        for row in self.rows:
            if row.candidate == candidate:
                return row
        raise KeyError(candidate)


def load_frontier_schedule(path: Path | None = None) -> FrontierScheduleReport:
    report_path = path or DEFAULT_SCHEDULE_PATH
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return FrontierScheduleReport(
        active_benchmark=str(data["active_benchmark"]),
        rows=tuple(
            ScheduleRow(
                action=str(row["action"]),
                bucket=str(row["bucket"]),
                candidate=str(row["candidate"]),
                priority=int(row["priority"]),
                reason=str(row["reason"]),
            )
            for row in data["rows"]
        ),
        seed_contract={
            str(role): SeedContractEntry(
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
                use=None if row.get("use") is None else str(row["use"]),
            )
            for role, row in data["seed_contract"].items()
        },
    )
