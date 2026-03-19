from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPLAY_PATH = Path("outputs/reports/portfolio_frontier_replay.json")


@dataclass(frozen=True)
class ReplayRow:
    case_candidate: str
    dev_delta_vs_round6: float
    dev_mean: float
    family: str
    guardrail_277: float
    sentinel_193: float
    support_233: float
    track: str
    verdict: str
    weakness_269: float
    weakness_alias: str


@dataclass(frozen=True)
class FrontierReplayReport:
    alias_map: dict[str, str]
    grouped: dict[str, tuple[str, ...]]
    rows: tuple[ReplayRow, ...]

    def row_by_candidate(self, candidate: str) -> ReplayRow:
        for row in self.rows:
            if row.case_candidate == candidate:
                return row
        raise KeyError(candidate)


def load_frontier_replay(path: Path | None = None) -> FrontierReplayReport:
    report_path = path or DEFAULT_REPLAY_PATH
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return FrontierReplayReport(
        alias_map={str(candidate): str(alias) for candidate, alias in data["alias_map"].items()},
        grouped={
            str(verdict): tuple(str(candidate) for candidate in candidates)
            for verdict, candidates in data["grouped"].items()
        },
        rows=tuple(
            ReplayRow(
                case_candidate=str(row["case_candidate"]),
                dev_delta_vs_round6=float(row["dev_delta_vs_round6"]),
                dev_mean=float(row["dev_mean"]),
                family=str(row["family"]),
                guardrail_277=float(row["guardrail_277"]),
                sentinel_193=float(row["sentinel_193"]),
                support_233=float(row["support_233"]),
                track=str(row["track"]),
                verdict=str(row["verdict"]),
                weakness_269=float(row["weakness_269"]),
                weakness_alias=str(row["weakness_alias"]),
            )
            for row in data["rows"]
        ),
    )
