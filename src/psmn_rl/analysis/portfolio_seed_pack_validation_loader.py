from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SEED_PACK_VALIDATION_PATH = Path("outputs/reports/portfolio_seed_pack_validation.json")


@dataclass(frozen=True)
class SeedPackValidationRow:
    candidate: str
    dev_delta_vs_round6: float | None
    dev_mean: float | None
    family: str
    guardrail_277: float
    policy_bucket: str
    screen_rule: str
    stage1_reason: str
    support_233: float | None
    tier: str
    track: str
    validation_bucket: str
    weakness_269: float


@dataclass(frozen=True)
class SeedPackValidationReport:
    needs_review: tuple[str, ...]
    rows: tuple[SeedPackValidationRow, ...]
    validated_reserve: tuple[str, ...]
    validated_restart_default: tuple[str, ...]
    validated_retired: tuple[str, ...]

    def row_by_candidate(self, candidate: str) -> SeedPackValidationRow:
        for row in self.rows:
            if row.candidate == candidate:
                return row
        raise KeyError(candidate)


def load_seed_pack_validation_report(path: Path | None = None) -> SeedPackValidationReport:
    report_path = path or DEFAULT_SEED_PACK_VALIDATION_PATH
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return SeedPackValidationReport(
        needs_review=tuple(str(value) for value in data["needs_review"]),
        rows=tuple(
            SeedPackValidationRow(
                candidate=str(row["candidate"]),
                dev_delta_vs_round6=(
                    None if row.get("dev_delta_vs_round6") is None else float(row["dev_delta_vs_round6"])
                ),
                dev_mean=None if row.get("dev_mean") is None else float(row["dev_mean"]),
                family=str(row["family"]),
                guardrail_277=float(row["guardrail_277"]),
                policy_bucket=str(row["policy_bucket"]),
                screen_rule=str(row["screen_rule"]),
                stage1_reason=str(row["stage1_reason"]),
                support_233=None if row.get("support_233") is None else float(row["support_233"]),
                tier=str(row["tier"]),
                track=str(row["track"]),
                validation_bucket=str(row["validation_bucket"]),
                weakness_269=float(row["weakness_269"]),
            )
            for row in data["rows"]
        ),
        validated_reserve=tuple(str(value) for value in data["validated_reserve"]),
        validated_restart_default=tuple(str(value) for value in data["validated_restart_default"]),
        validated_retired=tuple(str(value) for value in data["validated_retired"]),
    )
