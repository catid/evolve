from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DOCTOR_REPORT_PATH = Path("outputs/reports/portfolio_frontier_doctor.json")


@dataclass(frozen=True)
class DoctorCheck:
    label: str
    status: str
    detail: str


@dataclass(frozen=True)
class FrontierDoctorReport:
    overall: str
    checks: tuple[DoctorCheck, ...]

    def check_by_label(self, label: str) -> DoctorCheck:
        for check in self.checks:
            if check.label == label:
                return check
        raise KeyError(label)


def load_frontier_doctor_report(path: Path | None = None) -> FrontierDoctorReport:
    report_path = path or DEFAULT_DOCTOR_REPORT_PATH
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return FrontierDoctorReport(
        overall=str(data["overall"]),
        checks=tuple(
            DoctorCheck(
                label=str(row["label"]),
                status=str(row["status"]),
                detail=str(row["detail"]),
            )
            for row in data["checks"]
        ),
    )
