from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MANIFEST_PATH = Path("outputs/reports/portfolio_frontier_manifest.json")


@dataclass(frozen=True)
class ManifestRow:
    candidate: str
    manifest_bucket: str
    replay_verdict: str | None
    scorer_verdict: str | None


@dataclass(frozen=True)
class FrontierManifestReport:
    active: tuple[str, ...]
    default_restart: tuple[str, ...]
    hold_only: tuple[str, ...]
    replay_validated_alternates: tuple[str, ...]
    retired: tuple[str, ...]
    rows: tuple[ManifestRow, ...]
    seed_clean_unconfirmed: tuple[str, ...]

    def row_by_candidate(self, candidate: str) -> ManifestRow:
        for row in self.rows:
            if row.candidate == candidate:
                return row
        raise KeyError(candidate)


def load_frontier_manifest(path: Path | None = None) -> FrontierManifestReport:
    report_path = path or DEFAULT_MANIFEST_PATH
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return FrontierManifestReport(
        active=tuple(str(value) for value in data["active"]),
        default_restart=tuple(str(value) for value in data["default_restart"]),
        hold_only=tuple(str(value) for value in data["hold_only"]),
        replay_validated_alternates=tuple(str(value) for value in data["replay_validated_alternates"]),
        retired=tuple(str(value) for value in data["retired"]),
        rows=tuple(
            ManifestRow(
                candidate=str(row["candidate"]),
                manifest_bucket=str(row["manifest_bucket"]),
                replay_verdict=None if row["replay_verdict"] is None else str(row["replay_verdict"]),
                scorer_verdict=None if row["scorer_verdict"] is None else str(row["scorer_verdict"]),
            )
            for row in data["rows"]
        ),
        seed_clean_unconfirmed=tuple(str(value) for value in data["seed_clean_unconfirmed"]),
    )
