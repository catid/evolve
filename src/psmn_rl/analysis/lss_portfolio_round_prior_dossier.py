from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _float, _load_yaml, _read_json, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty

FRUITFUL_REPORT = Path("outputs/reports/portfolio_stage1_screening_fruitful.json")
STRESS_REPORT = Path("outputs/reports/successor_stress_extended_stage1_screening.json")
ROUND_PRIORS = ["round5", "round7", "round10"]
TRIAGE_SEEDS = [
    ("prospective_c", 193),
    ("prospective_f", 233),
    ("prospective_h", 269),
    ("prospective_h", 277),
]


def _fruitful_summary_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data = _read_json(FRUITFUL_REPORT)
    return list(data["rows"]), list(data["candidate_summaries"]), dict(data["round6_summary"])


def _stress_rows() -> list[dict[str, Any]]:
    data = _read_json(STRESS_REPORT)
    return list(data["rows"])


def _seed_value(rows: list[dict[str, Any]], candidate: str, lane: str, seed: int) -> float:
    for row in rows:
        if (
            str(row.get("candidate")) == candidate
            and str(row.get("lane")) == lane
            and int(row.get("seed", -1)) == seed
            and str(row.get("label", "kl_lss_sare")) == "kl_lss_sare"
        ):
            return _float(row["final_greedy_success"])
    raise KeyError(f"missing row for {candidate} {lane}/{seed}")


def _candidate_summary(summary_rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    for row in summary_rows:
        if str(row["candidate"]) == candidate:
            return row
    raise KeyError(candidate)


def _bucket(dev_delta: float, agg_steps: float, reference_steps: float) -> str:
    if dev_delta < 0.0:
        return "cheapest_but_below_incumbent"
    if abs(dev_delta) <= 1e-9 and agg_steps <= reference_steps + 1e-9:
        return "recommended_conservative_default"
    if abs(dev_delta) <= 1e-9:
        return "higher_cost_same_signal"
    return "unexpected"


def render_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    fruitful_rows, summary_rows, round6_summary = _fruitful_summary_rows()
    stress_rows = _stress_rows()
    round7_summary = _candidate_summary(summary_rows, "round7")
    reference_steps = _float(round7_summary["mean_aggregate_steps"])

    rows: list[dict[str, Any]] = []
    for candidate in ROUND_PRIORS:
        summary = _candidate_summary(summary_rows, candidate)
        seed_scores = {
            "prospective_c_193": _seed_value(fruitful_rows, candidate, "prospective_c", 193),
            "prospective_f_233": _seed_value(fruitful_rows, candidate, "prospective_f", 233),
            "prospective_h_269": _seed_value(stress_rows, candidate, "prospective_h", 269),
            "prospective_h_277": _seed_value(stress_rows, candidate, "prospective_h", 277),
        }
        rows.append(
            {
                "candidate": candidate,
                "portfolio_dev_mean": _float(summary["candidate_mean"]),
                "delta_vs_round6": _float(summary["delta_vs_round6"]),
                "delta_vs_token": _float(summary["candidate_minus_token"]),
                "delta_vs_single": _float(summary["candidate_minus_single"]),
                "mean_aggregate_steps": _float(summary["mean_aggregate_steps"]),
                "mean_fine_tune_steps": _float(summary["mean_fine_tune_steps"]),
                "stage1_reason": str(summary["stage1_reason"]),
                **seed_scores,
                "bucket": _bucket(
                    _float(summary["delta_vs_round6"]),
                    _float(summary["mean_aggregate_steps"]),
                    reference_steps,
                ),
            }
        )

    lines = [
        "# Portfolio Round-Prior Dossier",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- active incumbent portfolio dev mean: `{_float(round6_summary['sare_mean']):.4f}`",
        "",
        "## Round Prior Comparison",
        "",
        "| Candidate | Portfolio Dev Mean | Delta vs round6 | Delta vs token | Delta vs single | Mean Aggregate Steps | Mean Fine-Tune Steps | c/193 | f/233 | h/269 | h/277 | Bucket |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['candidate']}` | `{row['portfolio_dev_mean']:.4f}` | `{row['delta_vs_round6']:.4f}` | `{row['delta_vs_token']:.4f}` | `{row['delta_vs_single']:.4f}` | `{row['mean_aggregate_steps']:.1f}` | `{row['mean_fine_tune_steps']:.1f}` | `{row['prospective_c_193']:.4f}` | `{row['prospective_f_233']:.4f}` | `{row['prospective_h_269']:.4f}` | `{row['prospective_h_277']:.4f}` | `{row['bucket']}` |"
        )

    recommended = [row["candidate"] for row in rows if row["bucket"] == "recommended_conservative_default"]
    cheaper_below = [row["candidate"] for row in rows if row["bucket"] == "cheapest_but_below_incumbent"]
    higher_cost = [row["candidate"] for row in rows if row["bucket"] == "higher_cost_same_signal"]

    lines.extend(
        [
            "",
            "## Search Default",
            "",
            f"- recommended conservative default: `{recommended}`",
            f"- cheaper but below-incumbent priors: `{cheaper_below}`",
            f"- higher-cost same-signal priors: `{higher_cost}`",
            "",
            "## Interpretation",
            "",
            "- On the measured triage seeds, `round5`, `round7`, and `round10` are identical: all fail `prospective_c/193`, all preserve `prospective_f/233 = 1.0000`, and all solve `prospective_h/269` and `prospective_h/277` at `1.0000`.",
            "- The difference is outside that narrow surface. `round5` is cheaper, but it is also below the incumbent on the broader portfolio dev mean, so it should not be the default conservative restart.",
            "- `round7` and `round10` tie the incumbent on the broader portfolio dev mean, but `round7` does so at lower aggregate and fine-tune cost than `round10`.",
            "- That makes `round7` the clean conservative default if future bounded search restarts from the measured round-count priors.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "rows": rows,
                "round6_summary": round6_summary,
                "recommended": recommended,
                "cheaper_below": cheaper_below,
                "higher_cost": higher_cost,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a dossier over the clean measured round-count priors")
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
