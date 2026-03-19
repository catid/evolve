from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ACTIVE_STATE_DOCTOR_PATH = Path("outputs/reports/portfolio_active_state_doctor.json")


@dataclass(frozen=True)
class ActiveStateDoctorCheck:
    detail: str
    label: str
    status: str


@dataclass(frozen=True)
class ActiveStateDoctorReport:
    overall: str
    checks: tuple[ActiveStateDoctorCheck, ...]

    def check_by_label(self, label: str) -> ActiveStateDoctorCheck:
        for check in self.checks:
            if check.label == label:
                return check
        raise KeyError(label)


def load_active_state_doctor_report(path: Path | None = None) -> ActiveStateDoctorReport:
    report_path = path or DEFAULT_ACTIVE_STATE_DOCTOR_PATH
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return ActiveStateDoctorReport(
        overall=str(data["overall"]),
        checks=tuple(
            ActiveStateDoctorCheck(
                detail=str(check["detail"]),
                label=str(check["label"]),
                status=str(check["status"]),
            )
            for check in data["checks"]
        ),
    )
