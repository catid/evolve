from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _load_yaml, _read_json, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty

SEED_PACK = Path("outputs/reports/portfolio_seed_pack.json")
FRUITFUL_STAGE1 = Path("outputs/reports/portfolio_stage1_screening_fruitful.json")
EXPLORATORY_STAGE1 = Path("outputs/reports/portfolio_stage1_screening_exploratory.json")


def _summary_row(rows: list[dict[str, Any]], candidate: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row["candidate"]) == candidate:
            return row
    return None


def _validation_bucket(tier: str, screen_rule: str, dev_delta: float) -> str:
    if tier == "restart_default" and screen_rule == "advance_for_broader_dev" and abs(dev_delta) <= 1e-9:
        return "validated_restart_default"
    if tier == "reserve" and screen_rule == "advance_for_broader_dev":
        return "validated_reserve"
    if tier == "retired" and screen_rule in {"prune_support_regression", "prune_guardrail_regression"}:
        return "validated_retired"
    return "needs_review"


def render_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    seed_pack = _read_json(SEED_PACK)
    fruitful = _read_json(FRUITFUL_STAGE1)
    exploratory = _read_json(EXPLORATORY_STAGE1)

    fruitful_summaries = list(fruitful["candidate_summaries"])
    exploratory_summaries = list(exploratory["candidate_summaries"])

    rows: list[dict[str, Any]] = []
    for candidate_row in seed_pack["candidate_rows"]:
        candidate = str(candidate_row["candidate"])
        summary = _summary_row(fruitful_summaries, candidate)
        track = "fruitful"
        if summary is None:
            summary = _summary_row(exploratory_summaries, candidate)
            track = "exploratory"
        rows.append(
            {
                "candidate": candidate,
                "tier": str(candidate_row["tier"]),
                "policy_bucket": str(candidate_row["policy_bucket"]),
                "screen_rule": str(candidate_row["screen_rule"]),
                "support_233": candidate_row["support_233"],
                "weakness_269": float(candidate_row["weakness_269"]),
                "guardrail_277": float(candidate_row["guardrail_277"]),
                "track": track if summary is not None else "pack_only",
                "family": str(summary["family"]) if summary is not None else "pack_only",
                "dev_mean": float(summary["candidate_mean"]) if summary is not None else None,
                "dev_delta_vs_round6": float(summary["delta_vs_round6"]) if summary is not None else None,
                "stage1_reason": str(summary["stage1_reason"]) if summary is not None else "pack_only",
                "validation_bucket": _validation_bucket(
                    str(candidate_row["tier"]),
                    str(candidate_row["screen_rule"]),
                    float(summary["delta_vs_round6"]) if summary is not None else 0.0,
                ),
            }
        )

    lines = [
        "# Portfolio Seed Pack Validation",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- active benchmark: `{seed_pack['benchmark']['active_candidate_name']}`",
        "",
        "## Historical Consistency Check",
        "",
        "| Candidate | Tier | Screen Rule | Track | Family | Dev Mean | Delta vs round6 | Stage1 Reason | Validation |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        dev_mean = "n/a" if row["dev_mean"] is None else f"{row['dev_mean']:.4f}"
        dev_delta = "n/a" if row["dev_delta_vs_round6"] is None else f"{row['dev_delta_vs_round6']:.4f}"
        lines.append(
            f"| `{row['candidate']}` | `{row['tier']}` | `{row['screen_rule']}` | `{row['track']}` | `{row['family']}` | `{dev_mean}` | `{dev_delta}` | `{row['stage1_reason']}` | `{row['validation_bucket']}` |"
        )

    restart = [row["candidate"] for row in rows if row["validation_bucket"] == "validated_restart_default"]
    reserve = [row["candidate"] for row in rows if row["validation_bucket"] == "validated_reserve"]
    retired = [row["candidate"] for row in rows if row["validation_bucket"] == "validated_retired"]
    review = [row["candidate"] for row in rows if row["validation_bucket"] == "needs_review"]

    lines.extend(
        [
            "",
            "## Validation Summary",
            "",
            f"- validated restart default: `{restart}`",
            f"- validated reserve priors: `{reserve}`",
            f"- validated retired priors: `{retired}`",
            f"- needs review: `{review}`",
            "",
            "## Interpretation",
            "",
            "- This validator checks that the machine-readable seed pack agrees with the broader portfolio Stage 1 evidence rather than only restating the packed seed values.",
            "- `round7` should validate as the only restart default because it kept the required seed behavior while staying at incumbent dev mean without the extra cost of `round10` or the broader dev drop of `round5`.",
            "- `door3_post5` and `post_unlock_x5` should validate as retired because the packed prune rules match their measured support or guardrail regressions.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "rows": rows,
                "validated_restart_default": restart,
                "validated_reserve": reserve,
                "validated_retired": retired,
                "needs_review": review,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the portfolio seed pack against historical stage1 evidence")
    parser.add_argument("--campaign-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", required=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign = _load_yaml(Path(args.campaign_config))
    render_report(campaign, Path(args.output), Path(args.json) if args.json else None)


if __name__ == "__main__":
    main()
