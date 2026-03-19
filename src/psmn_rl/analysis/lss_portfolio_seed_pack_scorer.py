from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _load_yaml, _read_json, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty

SEED_PACK = Path("outputs/reports/portfolio_seed_pack.json")
FRUITFUL_STAGE1 = Path("outputs/reports/portfolio_stage1_screening_fruitful.json")
EXPLORATORY_STAGE1 = Path("outputs/reports/portfolio_stage1_screening_exploratory.json")


def _score_candidate(
    support_233: float | None,
    weakness_269: float,
    guardrail_277: float,
    incumbent_269: float,
    dev_delta_vs_round6: float | None,
) -> str:
    eps = 1e-9
    if support_233 is not None and support_233 < 1.0 - eps:
        return "prune_support_regression"
    if guardrail_277 < 1.0 - eps:
        return "prune_guardrail_regression"
    if weakness_269 < incumbent_269 - eps:
        return "prune_weakness_regression"
    if weakness_269 <= incumbent_269 + eps:
        return "hold_tie_only"
    if dev_delta_vs_round6 is None:
        return "needs_broader_dev_measurement"
    if dev_delta_vs_round6 < -eps:
        return "hold_seed_clean_but_below_incumbent"
    return "advance_for_broader_dev"


def _summary_map(*summary_sets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for summaries in summary_sets:
        for row in summaries:
            merged[str(row["candidate"])] = row
    return merged


def render_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    seed_pack = _read_json(SEED_PACK)
    fruitful = _read_json(FRUITFUL_STAGE1)
    exploratory = _read_json(EXPLORATORY_STAGE1)

    summary_rows = _summary_map(list(fruitful["candidate_summaries"]), list(exploratory["candidate_summaries"]))
    incumbent_269 = float(seed_pack["screening_rules"]["seed_roles"]["ranking_weakness"]["required_min_success"]) - 1e-9

    rows: list[dict[str, Any]] = []
    for row in seed_pack["candidate_rows"]:
        candidate = str(row["candidate"])
        summary = summary_rows.get(candidate)
        dev_delta = float(summary["delta_vs_round6"]) if summary is not None else None
        dev_mean = float(summary["candidate_mean"]) if summary is not None else None
        verdict = _score_candidate(
            float(row["support_233"]) if row["support_233"] is not None else None,
            float(row["weakness_269"]),
            float(row["guardrail_277"]),
            incumbent_269,
            dev_delta,
        )
        rows.append(
            {
                "candidate": candidate,
                "tier": str(row["tier"]),
                "policy_bucket": str(row["policy_bucket"]),
                "support_233": row["support_233"],
                "weakness_269": float(row["weakness_269"]),
                "guardrail_277": float(row["guardrail_277"]),
                "dev_mean": dev_mean,
                "dev_delta_vs_round6": dev_delta,
                "verdict": verdict,
            }
        )

    lines = [
        "# Portfolio Seed Pack Scorer",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- active benchmark: `{seed_pack['benchmark']['active_candidate_name']}`",
        "",
        "## Candidate Verdicts",
        "",
        "| Candidate | Tier | support_233 | weakness_269 | guardrail_277 | Dev Mean | Delta vs round6 | Verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        support = "n/a" if row["support_233"] is None else f"{float(row['support_233']):.4f}"
        dev_mean = "n/a" if row["dev_mean"] is None else f"{float(row['dev_mean']):.4f}"
        dev_delta = "n/a" if row["dev_delta_vs_round6"] is None else f"{float(row['dev_delta_vs_round6']):.4f}"
        lines.append(
            f"| `{row['candidate']}` | `{row['tier']}` | `{support}` | `{row['weakness_269']:.4f}` | `{row['guardrail_277']:.4f}` | `{dev_mean}` | `{dev_delta}` | `{row['verdict']}` |"
        )

    grouped = {
        verdict: [row["candidate"] for row in rows if row["verdict"] == verdict]
        for verdict in sorted({str(row["verdict"]) for row in rows})
    }

    lines.extend(
        [
            "",
            "## Verdict Groups",
            "",
            f"- groups: `{grouped}`",
            "",
            "## Interpretation",
            "",
            "- This scorer is the executable pre-fairness gate for future bounded mini-sweeps built on the portfolio seed pack.",
            "- It prunes support, guardrail, and weakness regressions immediately; it holds exact weakness ties; and it only advances seed-clean lines that also avoid a broader dev drop when that dev summary is available.",
            "- On the current measured frontier, `round7` and `round10` should advance for broader dev, `round5` should be held as seed-clean but below-incumbent, and the retired lines should be pruned.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "rows": rows,
                "grouped": grouped,
                "incumbent_269": incumbent_269,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score the measured frontier with the portfolio seed-pack rules")
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
