from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_GATE_REPORT_PATH = Path("outputs/reports/portfolio_gate_report.json")


@dataclass(frozen=True)
class GateCheck:
    detail: str
    name: str
    status: str


@dataclass(frozen=True)
class PortfolioGateReport:
    candidate_pack: str
    candidate_pack_validation: tuple[GateCheck, ...]
    checks: tuple[GateCheck, ...]
    frozen_pack: str
    frozen_pack_validation: tuple[GateCheck, ...]
    mode: str
    verdict: str

    def check_by_name(self, name: str) -> GateCheck:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)


def _load_checks(rows: list[dict[str, object]]) -> tuple[GateCheck, ...]:
    return tuple(
        GateCheck(
            detail=str(row["detail"]),
            name=str(row["name"]),
            status=str(row["status"]),
        )
        for row in rows
    )


def load_portfolio_gate_report(path: Path | None = None) -> PortfolioGateReport:
    report_path = path or DEFAULT_GATE_REPORT_PATH
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return PortfolioGateReport(
        candidate_pack=str(data["candidate_pack"]),
        candidate_pack_validation=_load_checks(data["candidate_pack_validation"]),
        checks=_load_checks(data["checks"]),
        frozen_pack=str(data["frozen_pack"]),
        frozen_pack_validation=_load_checks(data["frozen_pack_validation"]),
        mode=str(data["mode"]),
        verdict=str(data["verdict"]),
    )
