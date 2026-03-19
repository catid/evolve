from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DOCS_AUDIT_PATH = Path("outputs/reports/portfolio_frontier_docs_audit.json")


@dataclass(frozen=True)
class DocsAuditCheck:
    path: str
    status: str
    missing: tuple[str, ...]


@dataclass(frozen=True)
class FrontierDocsAuditReport:
    overall: str
    active_candidate: str
    active_candidate_pack: str
    default_restart_prior: str
    replay_validated_alternate: str
    checks: tuple[DocsAuditCheck, ...]

    def check_by_path(self, path: str) -> DocsAuditCheck:
        for check in self.checks:
            if check.path == path:
                return check
        raise KeyError(path)


def load_frontier_docs_audit_report(path: Path | None = None) -> FrontierDocsAuditReport:
    report_path = path or DEFAULT_DOCS_AUDIT_PATH
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return FrontierDocsAuditReport(
        overall=str(data["overall"]),
        active_candidate=str(data["active_candidate"]),
        active_candidate_pack=str(data["active_candidate_pack"]),
        default_restart_prior=str(data["default_restart_prior"]),
        replay_validated_alternate=str(data["replay_validated_alternate"]),
        checks=tuple(
            DocsAuditCheck(
                path=str(row["path"]),
                status=str(row["status"]),
                missing=tuple(str(value) for value in row["missing"]),
            )
            for row in data["checks"]
        ),
    )
