from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SEED_PACK_SCORER_PATH = Path("outputs/reports/portfolio_seed_pack_scorer.json")


@dataclass(frozen=True)
class SeedPackScorerRow:
    candidate: str
    dev_delta_vs_round6: float | None
    dev_mean: float | None
    guardrail_277: float
    policy_bucket: str
    support_233: float | None
    tier: str
    verdict: str
    weakness_269: float


@dataclass(frozen=True)
class SeedPackScorerReport:
    grouped: dict[str, tuple[str, ...]]
    incumbent_269: float
    rows: tuple[SeedPackScorerRow, ...]

    def row_by_candidate(self, candidate: str) -> SeedPackScorerRow:
        for row in self.rows:
            if row.candidate == candidate:
                return row
        raise KeyError(candidate)


def load_seed_pack_scorer_report(path: Path | None = None) -> SeedPackScorerReport:
    report_path = path or DEFAULT_SEED_PACK_SCORER_PATH
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return SeedPackScorerReport(
        grouped={
            str(verdict): tuple(str(candidate) for candidate in candidates)
            for verdict, candidates in data["grouped"].items()
        },
        incumbent_269=float(data["incumbent_269"]),
        rows=tuple(
            SeedPackScorerRow(
                candidate=str(row["candidate"]),
                dev_delta_vs_round6=(
                    None if row.get("dev_delta_vs_round6") is None else float(row["dev_delta_vs_round6"])
                ),
                dev_mean=None if row.get("dev_mean") is None else float(row["dev_mean"]),
                guardrail_277=float(row["guardrail_277"]),
                policy_bucket=str(row["policy_bucket"]),
                support_233=None if row.get("support_233") is None else float(row["support_233"]),
                tier=str(row["tier"]),
                verdict=str(row["verdict"]),
                weakness_269=float(row["weakness_269"]),
            )
            for row in data["rows"]
        ),
    )
