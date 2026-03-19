from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _load_yaml, _read_json, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty

ROUND_PRIOR_DOSSIER = Path("outputs/reports/portfolio_round_prior_dossier.json")
TRIAGE_MATRIX = Path("outputs/reports/portfolio_triage_matrix.json")
STRUCTURAL_PROBE = Path("outputs/reports/portfolio_structural_probe.json")
FAIRNESS_REPORT = Path("outputs/reports/portfolio_stage3_fairness.json")

TARGET_ORDER = [
    "round6",
    "round7",
    "round10",
    "round5",
    "door3_post5",
    "post_unlock_x5",
]


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _policy_bucket(
    candidate: str,
    triage_bucket: str | None,
    dossier_bucket: str | None,
    support_status: str | None,
) -> str:
    if candidate == "round6":
        return "active_incumbent"
    if candidate == "round7" and triage_bucket == "conservative_clean_prior" and dossier_bucket == "recommended_conservative_default":
        return "restart_default"
    if candidate == "round10" and triage_bucket == "conservative_clean_prior" and dossier_bucket == "higher_cost_same_signal":
        return "reserve_same_signal_higher_cost"
    if candidate == "round5" and triage_bucket == "conservative_clean_prior" and dossier_bucket == "cheapest_but_below_incumbent":
        return "reserve_below_incumbent"
    if candidate == "door3_post5" and support_status == "measured_support_regression":
        return "retire_structural_regression"
    if candidate == "post_unlock_x5" and triage_bucket in {"partial_guardrail_loss", "local_only_fix"}:
        return "retire_local_only_fix"
    return "unresolved"


def _row_by_candidate(rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    for row in rows:
        if str(row["candidate"]) == candidate:
            return row
    raise KeyError(candidate)


def _probe_score(detail_rows: list[dict[str, Any]], candidate: str, lane: str, seed: int) -> float | None:
    key = f"{lane}_{seed}"
    candidate_key = {
        "round6": "round6",
        "round7": "round7",
        "door3_post5": "door3_post5",
        "token_dense": "token_dense",
        "single_expert": "single_expert",
    }.get(candidate)
    if candidate_key is None:
        return None
    for row in detail_rows:
        if str(row["lane"]) == lane and int(row["seed"]) == seed:
            return float(row[candidate_key])
    raise KeyError(key)


def render_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    round_dossier = _read_json(ROUND_PRIOR_DOSSIER)
    triage = _read_json(TRIAGE_MATRIX)
    structural = _read_json(STRUCTURAL_PROBE)
    fairness = _read_json(FAIRNESS_REPORT)

    dossier_rows = list(round_dossier["rows"])
    triage_rows = list(triage["rows"])
    structural_rows = list(structural["detail_rows"])
    support_status = str(structural["support_status"])
    surviving_candidates = list(fairness["surviving_candidates"])

    rows: list[dict[str, Any]] = []
    for candidate in TARGET_ORDER:
        triage_row = _row_by_candidate(triage_rows, candidate)
        dossier_row = None
        if candidate != "round6":
            try:
                dossier_row = _row_by_candidate(dossier_rows, candidate)
            except KeyError:
                dossier_row = None
        rows.append(
            {
                "candidate": candidate,
                "triage_bucket": str(triage_row["bucket"]),
                "dossier_bucket": str(dossier_row["bucket"]) if dossier_row is not None else "n/a",
                "support_233": _probe_score(structural_rows, candidate, "prospective_f", 233)
                if _probe_score(structural_rows, candidate, "prospective_f", 233) is not None
                else triage_row["benchmark_support_233"],
                "weakness_269": triage_row["bounded_weakness_269"],
                "guardrail_277": triage_row["guardrail_277"],
                "policy_bucket": _policy_bucket(
                    candidate,
                    str(triage_row["bucket"]),
                    str(dossier_row["bucket"]) if dossier_row is not None else None,
                    support_status,
                ),
            }
        )

    restart_default = [row["candidate"] for row in rows if row["policy_bucket"] == "restart_default"]
    reserve_priors = [
        row["candidate"]
        for row in rows
        if row["policy_bucket"] in {"reserve_same_signal_higher_cost", "reserve_below_incumbent"}
    ]
    retired_priors = [
        row["candidate"]
        for row in rows
        if row["policy_bucket"] in {"retire_structural_regression", "retire_local_only_fix"}
    ]

    lines = [
        "# Portfolio Restart Policy",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- stage3 surviving challengers: `{surviving_candidates}`",
        f"- structural support status: `{support_status}`",
        "",
        "## Restart Frontier",
        "",
        "| Candidate | Support 233 | Weakness 269 | Guardrail 277 | Triage Bucket | Dossier Bucket | Policy Bucket |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['candidate']}` | `{_fmt(row['support_233'])}` | `{_fmt(row['weakness_269'])}` | `{_fmt(row['guardrail_277'])}` | `{row['triage_bucket']}` | `{row['dossier_bucket']}` | `{row['policy_bucket']}` |"
        )

    lines.extend(
        [
            "",
            "## Search Policy",
            "",
            f"- active incumbent: `{[row['candidate'] for row in rows if row['policy_bucket'] == 'active_incumbent']}`",
            f"- bounded restart default: `{restart_default}`",
            f"- reserve priors: `{reserve_priors}`",
            f"- retired priors: `{retired_priors}`",
            "",
            "## Interpretation",
            "",
            "- The post-control field is still empty, so future bounded search should restart from measured priors rather than pretend a live challenger already exists.",
            "- `round7` is the only clean restart default: it preserves the measured support seed, fixes the bounded weakness seed, preserves the guardrail seed, and does so without the higher cost of `round10` or the below-incumbent broader dev mean of `round5`.",
            "- `door3_post5` should be retired as a fallback prior, not kept in reserve. The direct probe measured a support regression on `prospective_f/233`, so it is no longer merely unmeasured.",
            "- `post_unlock_x5` should also stay retired as a local-only fix because it gives back guardrail performance on `prospective_h/277` even though it can solve the bounded weakness seed.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "rows": rows,
                "restart_default": restart_default,
                "reserve_priors": reserve_priors,
                "retired_priors": retired_priors,
                "support_status": support_status,
                "surviving_candidates": surviving_candidates,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a restart policy from the measured portfolio frontier")
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
