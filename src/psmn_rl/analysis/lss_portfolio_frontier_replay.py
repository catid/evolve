from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_portfolio_seed_pack_scorer import _score_candidate
from psmn_rl.analysis.lss_post_pass_campaign import _load_yaml, _read_json, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty

CASEBOOK = Path("outputs/reports/portfolio_hard_seed_casebook.json")
WEAKNESS_LEADERBOARD = Path("outputs/reports/portfolio_weakness_leaderboard.json")
FRUITFUL_STAGE1 = Path("outputs/reports/portfolio_stage1_screening_fruitful.json")
EXPLORATORY_STAGE1 = Path("outputs/reports/portfolio_stage1_screening_exploratory.json")
SEED_PACK = Path("outputs/reports/portfolio_seed_pack.json")

ALIAS_MAP = {
    "round10": "round10",
    "round10_carry2_post4": "carry2_post4",
    "round10_door2_post4": "door2_post4",
    "round10_post_unlock_x5": "post_unlock_x5",
}


def _summary_map(*summary_sets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for summaries in summary_sets:
        for row in summaries:
            merged[str(row["candidate"])] = row
    return merged


def _casebook_seed(case_rows: list[dict[str, Any]], candidate: str, lane: str, seed: int) -> float | None:
    for row in case_rows:
        if str(row["candidate"]) == candidate and str(row["lane"]) == lane and int(row["seed"]) == seed:
            return float(row["sare"])
    return None


def _weakness_entry(entries: list[dict[str, Any]], candidate: str) -> dict[str, Any] | None:
    for entry in entries:
        if str(entry["candidate"]) == candidate:
            return entry
    return None


def render_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    casebook = _read_json(CASEBOOK)
    weakness = _read_json(WEAKNESS_LEADERBOARD)
    fruitful = _read_json(FRUITFUL_STAGE1)
    exploratory = _read_json(EXPLORATORY_STAGE1)
    seed_pack = _read_json(SEED_PACK)

    summary_rows = _summary_map(list(fruitful["candidate_summaries"]), list(exploratory["candidate_summaries"]))
    incumbent_269 = float(seed_pack["screening_rules"]["seed_roles"]["ranking_weakness"]["required_min_success"]) - 1e-9

    case_rows = list(casebook["seed_rows"])
    weakness_entries = list(weakness["entries"])

    rows: list[dict[str, Any]] = []
    for case_candidate, weakness_alias in ALIAS_MAP.items():
        weak = _weakness_entry(weakness_entries, weakness_alias)
        if weak is None:
            continue
        summary = summary_rows.get(case_candidate)
        support_233 = _casebook_seed(case_rows, case_candidate, "prospective_f", 233)
        sentinel_193 = _casebook_seed(case_rows, case_candidate, "prospective_c", 193)
        weakness_269 = float(weak["seed_269"])
        guardrail_277 = float(weak["seed_277"]) if weak["seed_277"] is not None else None
        verdict = _score_candidate(
            support_233,
            weakness_269,
            float(guardrail_277) if guardrail_277 is not None else 0.0,
            incumbent_269,
            float(summary["delta_vs_round6"]) if summary is not None else None,
        )
        rows.append(
            {
                "case_candidate": case_candidate,
                "weakness_alias": weakness_alias,
                "support_233": support_233,
                "sentinel_193": sentinel_193,
                "weakness_269": weakness_269,
                "guardrail_277": guardrail_277,
                "dev_mean": float(summary["candidate_mean"]) if summary is not None else None,
                "dev_delta_vs_round6": float(summary["delta_vs_round6"]) if summary is not None else None,
                "track": str(summary["track"]) if summary is not None else "unknown",
                "family": str(summary["family"]) if summary is not None else "unknown",
                "verdict": verdict,
            }
        )

    lines = [
        "# Portfolio Frontier Replay",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "- replay surface: candidates with measured `prospective_f/233` support plus historical `prospective_h/269/277` weakness coverage",
        "",
        "## Measured Quartet Replay",
        "",
        "| Candidate | Weakness Alias | Track | Family | c/193 | f/233 | h/269 | h/277 | Dev Mean | Delta vs round6 | Verdict |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        dev_mean = "n/a" if row["dev_mean"] is None else f"{row['dev_mean']:.4f}"
        dev_delta = "n/a" if row["dev_delta_vs_round6"] is None else f"{row['dev_delta_vs_round6']:.4f}"
        guardrail = "n/a" if row["guardrail_277"] is None else f"{float(row['guardrail_277']):.4f}"
        lines.append(
            f"| `{row['case_candidate']}` | `{row['weakness_alias']}` | `{row['track']}` | `{row['family']}` | `{float(row['sentinel_193']):.4f}` | `{float(row['support_233']):.4f}` | `{float(row['weakness_269']):.4f}` | `{guardrail}` | `{dev_mean}` | `{dev_delta}` | `{row['verdict']}` |"
        )

    grouped = {
        verdict: [row["case_candidate"] for row in rows if row["verdict"] == verdict]
        for verdict in sorted({str(row["verdict"]) for row in rows})
    }
    lines.extend(
        [
            "",
            "## Replay Summary",
            "",
            f"- verdict groups: `{grouped}`",
            "",
            "## Interpretation",
            "",
            "- This replay scores only the historically measured quartet, not every historical candidate name, because most lines never received the full support-plus-weakness measurement surface.",
            "- `round10` is the only fully measured historical line that remains seed-clean enough to deserve broader dev spending if the project reopens bounded search from the near-neighbor side.",
            "- `round10_carry2_post4`, `round10_door2_post4`, and `round10_post_unlock_x5` all fail the `prospective_h/277` guardrail once the historical weakness surface is stitched in, so they should stay pruned despite looking clean on the narrower portfolio dev split alone.",
            "- `prospective_c/193` stays all-zero across the replay surface, so it remains a sentinel rather than a ranking seed.",
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
                "alias_map": ALIAS_MAP,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay the portfolio seed-pack rules on the historically measured quartet")
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
