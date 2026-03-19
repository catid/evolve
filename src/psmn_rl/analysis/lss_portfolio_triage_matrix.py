from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _float, _load_yaml, _read_json, _write_json
from psmn_rl.analysis.lss_portfolio_weakness_leaderboard import REPORT_SOURCES, _load_rows, _seed_row
from psmn_rl.utils.io import get_git_commit, get_git_dirty

TARGET_SEEDS = [
    ("prospective_c", 193, "global_hard_sentinel"),
    ("prospective_f", 233, "benchmark_support"),
    ("prospective_h", 269, "bounded_weakness"),
    ("prospective_h", 277, "same_lane_guardrail"),
]

PRIOR_VARIANTS = [
    ("round6", "active"),
    ("round5", "mixed_round_prior"),
    ("round7", "mixed_round_prior"),
    ("round10", "mixed_round_prior"),
    ("door3_post5", "migration_only"),
    ("post_unlock_x5", "migration_only"),
]

LABEL_DIRS = {
    "sare": "kl_lss_sare",
    "token_dense": "kl_lss_token_dense",
    "single_expert": "kl_lss_single_expert",
}

REPORT_SOURCES_EXT = {
    **REPORT_SOURCES,
    "portfolio_fruitful": "outputs/reports/portfolio_stage1_screening_fruitful.json",
}

CANDIDATE_SOURCE_MAP = {
    "round5": {
        ("prospective_c", 193): "portfolio_fruitful",
        ("prospective_f", 233): "portfolio_fruitful",
        ("prospective_h", 269): "stress_extended",
        ("prospective_h", 277): "stress_extended",
    },
    "round7": {
        ("prospective_c", 193): "portfolio_fruitful",
        ("prospective_f", 233): "portfolio_fruitful",
        ("prospective_h", 269): "stress_extended",
        ("prospective_h", 277): "stress_extended",
    },
    "round10": {
        ("prospective_c", 193): "portfolio_fruitful",
        ("prospective_f", 233): "portfolio_fruitful",
        ("prospective_h", 269): "stress_extended",
        ("prospective_h", 277): "stress_extended",
    },
    "door3_post5": {
        ("prospective_h", 269): "migration",
        ("prospective_h", 277): "migration",
    },
    "post_unlock_x5": {
        ("prospective_h", 269): "migration",
        ("prospective_h", 277): "migration",
    },
}


def _active_summary(campaign: dict[str, Any], lane: str, seed: int, variant: str) -> dict[str, Any]:
    if variant == "sare":
        root = Path(campaign["current_round6_sare_roots"][lane])
    else:
        root = Path(campaign["current_round6_control_roots"][lane])
    return _read_json(root / f"seed_{seed}" / LABEL_DIRS[variant] / "summary.json")


def _prior_bucket(seed_233: float | None, seed_269: float, seed_277: float, incumbent_269: float, structural: bool) -> str:
    if seed_269 > incumbent_269 and seed_277 >= 1.0 - 1e-9 and seed_233 is None:
        return "support_unmeasured_structural" if structural else "support_unmeasured_conservative"
    if seed_269 > incumbent_269 and seed_233 is not None and seed_233 >= 1.0 - 1e-9 and seed_277 >= 1.0 - 1e-9:
        return "structural_clean_prior" if structural else "conservative_clean_prior"
    if seed_269 > incumbent_269 and seed_233 is not None and seed_233 >= 1.0 - 1e-9 and seed_277 >= 0.85:
        return "partial_guardrail_loss"
    if seed_269 > incumbent_269:
        return "local_only_fix"
    return "not_better_than_incumbent"


def _candidate_seed_scores(campaign: dict[str, Any], candidate: str, source: str) -> dict[tuple[str, int], float | None]:
    if source == "active":
        result: dict[tuple[str, int], float | None] = {}
        for lane, seed, _role in TARGET_SEEDS:
            result[(lane, seed)] = _float(_active_summary(campaign, lane, seed, "sare")["final_greedy_success"])
        return result

    if source == "mixed_round_prior":
        result: dict[tuple[str, int], float | None] = {}
        for lane, seed, _role in TARGET_SEEDS:
            report_name = CANDIDATE_SOURCE_MAP[candidate].get((lane, seed))
            if report_name is None:
                result[(lane, seed)] = None
                continue
            rows = _load_rows(Path(REPORT_SOURCES_EXT[report_name]))
            row = _seed_row(rows, candidate, "kl_lss_sare", lane, seed)
            result[(lane, seed)] = _float(row["final_greedy_success"]) if row is not None else None
        return result

    if source == "migration_only":
        result = {}
        for lane, seed, _role in TARGET_SEEDS:
            report_name = CANDIDATE_SOURCE_MAP[candidate].get((lane, seed))
            if report_name is None:
                result[(lane, seed)] = None
                continue
            rows = _load_rows(Path(REPORT_SOURCES_EXT[report_name]))
            row = _seed_row(rows, candidate, "kl_lss_sare", lane, seed)
            result[(lane, seed)] = _float(row["final_greedy_success"]) if row is not None else None
        return result

    rows = _load_rows(Path(REPORT_SOURCES_EXT[source]))
    result: dict[tuple[str, int], float | None] = {}
    for lane, seed, _role in TARGET_SEEDS:
        row = _seed_row(rows, candidate, "kl_lss_sare", lane, seed)
        result[(lane, seed)] = _float(row["final_greedy_success"]) if row is not None else None
    return result


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def render_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    incumbent_269 = _float(_active_summary(campaign, "prospective_h", 269, "sare")["final_greedy_success"])
    token_scores = {(lane, seed): _float(_active_summary(campaign, lane, seed, "token_dense")["final_greedy_success"]) for lane, seed, _role in TARGET_SEEDS}
    single_scores = {(lane, seed): _float(_active_summary(campaign, lane, seed, "single_expert")["final_greedy_success"]) for lane, seed, _role in TARGET_SEEDS}

    rows: list[dict[str, Any]] = []
    for candidate, source in PRIOR_VARIANTS:
        scores = _candidate_seed_scores(campaign, candidate, source)
        seed_233 = scores[("prospective_f", 233)]
        seed_269 = _float(scores[("prospective_h", 269)] or 0.0)
        seed_277 = _float(scores[("prospective_h", 277)] or 0.0)
        rows.append(
            {
                "candidate": candidate,
                "source": source,
                "global_hard_193": scores[("prospective_c", 193)],
                "benchmark_support_233": seed_233,
                "bounded_weakness_269": seed_269,
                "guardrail_277": seed_277,
                "delta_vs_round6_269": seed_269 - incumbent_269,
                "bucket": _prior_bucket(seed_233, seed_269, seed_277, incumbent_269, structural=(not candidate.startswith("round") and candidate != "round6")),
            }
        )

    lines = [
        "# Portfolio Triage Matrix",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- target seeds: `{[f'{lane}/{seed}:{role}' for lane, seed, role in TARGET_SEEDS]}`",
        "",
        "## Active Controls On The Target Seeds",
        "",
        "| Variant | prospective_c/193 | prospective_f/233 | prospective_h/269 | prospective_h/277 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| `round6` | `{_float(_active_summary(campaign, 'prospective_c', 193, 'sare')['final_greedy_success']):.4f}` | `{_float(_active_summary(campaign, 'prospective_f', 233, 'sare')['final_greedy_success']):.4f}` | `{incumbent_269:.4f}` | `{_float(_active_summary(campaign, 'prospective_h', 277, 'sare')['final_greedy_success']):.4f}` |",
        f"| `token_dense` | `{token_scores[('prospective_c', 193)]:.4f}` | `{token_scores[('prospective_f', 233)]:.4f}` | `{token_scores[('prospective_h', 269)]:.4f}` | `{token_scores[('prospective_h', 277)]:.4f}` |",
        f"| `single_expert` | `{single_scores[('prospective_c', 193)]:.4f}` | `{single_scores[('prospective_f', 233)]:.4f}` | `{single_scores[('prospective_h', 269)]:.4f}` | `{single_scores[('prospective_h', 277)]:.4f}` |",
        "",
        "## Prior Matrix",
        "",
        "| Candidate | Source | c/193 | f/233 | h/269 | h/277 | delta vs round6 on 269 | Bucket |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['candidate']}` | `{row['source']}` | `{_fmt(row['global_hard_193'])}` | `{_fmt(row['benchmark_support_233'])}` | `{row['bounded_weakness_269']:.4f}` | `{row['guardrail_277']:.4f}` | `{row['delta_vs_round6_269']:.4f}` | `{row['bucket']}` |"
        )

    conservative = [row["candidate"] for row in rows if row["bucket"] == "conservative_clean_prior"]
    structural = [row["candidate"] for row in rows if row["bucket"] == "structural_clean_prior"]
    partial = [row["candidate"] for row in rows if row["bucket"] == "partial_guardrail_loss"]
    local = [row["candidate"] for row in rows if row["bucket"] == "local_only_fix"]
    support_unmeasured = [row["candidate"] for row in rows if row["bucket"] in {"support_unmeasured_structural", "support_unmeasured_conservative"}]
    not_better = [row["candidate"] for row in rows if row["bucket"] == "not_better_than_incumbent"]

    lines.extend(
        [
            "",
            "## Search Use",
            "",
            f"- conservative clean priors: `{conservative}`",
            f"- structural clean priors: `{structural}`",
            f"- partial guardrail-loss priors: `{partial}`",
            f"- support-unmeasured but same-lane-clean priors: `{support_unmeasured}`",
            f"- local-only fixes: `{local}`",
            f"- not better than incumbent on the weakness seed: `{not_better}`",
            "",
            "## Interpretation",
            "",
            "- Every shortlisted prior still fails the global-hard sentinel `prospective_c/193`, so that seed remains a hardness sentinel rather than a near-term optimization target.",
            "- The useful separator is the pair `prospective_f/233` plus `prospective_h/269/277`: good priors preserve the benchmark-support seed when it is measured, improve the bounded weakness seed above `round6`, and avoid degrading the same-lane guardrail.",
            "- On that measured triage surface, `round5`, `round7`, and `round10` are the clean conservative starting points; `door3_post5` remains a same-lane-clean structural fallback but still needs direct `233` measurement before being treated as equally safe; and `post_unlock_x5` is only a partial revisit because it gives back some of the `277` guardrail.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "rows": rows,
                "token_scores": {f"{lane}:{seed}": value for (lane, seed), value in token_scores.items()},
                "single_scores": {f"{lane}:{seed}": value for (lane, seed), value in single_scores.items()},
                "incumbent_269": incumbent_269,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a triage matrix over the only seeds that matter for bounded portfolio follow-up")
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
