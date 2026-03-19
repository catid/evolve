from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SEED_PACK_DOCTOR_PATH = Path("outputs/reports/portfolio_seed_pack_doctor.json")


@dataclass(frozen=True)
class SeedPackDoctorCheck:
    detail: str
    label: str
    status: str


@dataclass(frozen=True)
class SeedPackDoctorReport:
    overall: str
    checks: tuple[SeedPackDoctorCheck, ...]

    def check_by_label(self, label: str) -> SeedPackDoctorCheck:
        for check in self.checks:
            if check.label == label:
                return check
        raise KeyError(label)


def load_seed_pack_doctor_report(path: Path | None = None) -> SeedPackDoctorReport:
    report_path = path or DEFAULT_SEED_PACK_DOCTOR_PATH
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return SeedPackDoctorReport(
        overall=str(data["overall"]),
        checks=tuple(
            SeedPackDoctorCheck(
                detail=str(check["detail"]),
                label=str(check["label"]),
                status=str(check["status"]),
            )
            for check in data["checks"]
        ),
    )
