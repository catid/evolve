from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _float, _load_yaml, _read_json, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty

TARGET_ROWS = [
    ("prospective_c", 193),
    ("prospective_f", 233),
]


def _round6_row(campaign: dict[str, Any], lane: str, seed: int, variant: str) -> float:
    if variant == "sare":
        root = Path(campaign["current_round6_sare_roots"][lane])
        label = "kl_lss_sare"
    else:
        root = Path(campaign["current_round6_control_roots"][lane])
        label = f"kl_lss_{variant}"
    data = _read_json(root / f"seed_{seed}" / label / "summary.json")
    return _float(data["final_greedy_success"])


def _candidate_row(stage1: dict[str, Any], candidate: str, lane: str, seed: int) -> float:
    for row in stage1["rows"]:
        if (
            str(row["candidate"]) == candidate
            and str(row["lane"]) == lane
            and int(row["seed"]) == seed
            and str(row["label"]) == "kl_lss_sare"
        ):
            return _float(row["final_greedy_success"])
    raise KeyError((candidate, lane, seed))


def _support_status(door3: float, round7: float) -> str:
    if door3 >= 1.0 - 1e-9 and round7 >= 1.0 - 1e-9:
        return "measured_clean_support"
    if door3 + 1e-9 < round7:
        return "measured_support_regression"
    return "measured_partial_support"


def render_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    stage1 = _read_json(Path(campaign["reports"]["stage1_json"]))
    detail_rows: list[dict[str, Any]] = []
    for lane, seed in TARGET_ROWS:
        detail_rows.append(
            {
                "lane": lane,
                "seed": seed,
                "round6": _round6_row(campaign, lane, seed, "sare"),
                "round7": _candidate_row(stage1, "round7", lane, seed),
                "door3_post5": _candidate_row(stage1, "door3_post5", lane, seed),
                "token_dense": _round6_row(campaign, lane, seed, "token_dense"),
                "single_expert": _round6_row(campaign, lane, seed, "single_expert"),
            }
        )

    c193 = next(row for row in detail_rows if row["lane"] == "prospective_c")
    f233 = next(row for row in detail_rows if row["lane"] == "prospective_f")
    support_status = _support_status(f233["door3_post5"], f233["round7"])

    lines = [
        "# Portfolio Structural Fallback Probe",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- support status: `{support_status}`",
        "",
        "## Seed Comparison",
        "",
        "| Lane | Seed | round6 | round7 | door3_post5 | token_dense | single_expert |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in detail_rows:
        lines.append(
            f"| `{row['lane']}` | {row['seed']} | `{row['round6']:.4f}` | `{row['round7']:.4f}` | `{row['door3_post5']:.4f}` | `{row['token_dense']:.4f}` | `{row['single_expert']:.4f}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `prospective_c/193` remains a global-hard sentinel. If all compared lines are still `0.0000`, this probe does not reopen that seed as a near-term optimization target.",
            "- `prospective_f/233` is the decisive measurement gap for the structural fallback. If `door3_post5` matches `round7` and `round6` there at `1.0000`, the structural fallback is now measured cleanly on the support seed instead of merely inferred from the weakness lane.",
            "- Even if `door3_post5` measures cleanly on `233`, it still remains a structural fallback rather than the new conservative default unless it offers some advantage beyond what the cheaper `round7` prior already provides.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "detail_rows": detail_rows,
                "support_status": support_status,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a direct structural-fallback probe on the measured triage seeds")
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
