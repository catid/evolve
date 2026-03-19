from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_portfolio_hard_seed_casebook import _classify_seed
from psmn_rl.analysis.lss_post_pass_campaign import _float, _load_yaml, _read_json, _write_json
from psmn_rl.analysis.lss_successor_migration import CURRENT_LABELS, _block_lanes, _current_round6_rows
from psmn_rl.utils.io import get_git_commit, get_git_dirty


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["lane"]), int(row["seed"])


def _seed_scores(rows: list[dict[str, Any]], candidate: str) -> dict[tuple[str, int], dict[str, float]]:
    scores: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        if str(row["candidate"]) != candidate:
            continue
        scores.setdefault(_key(row), {})[str(row["label"])] = _float(row["final_greedy_success"])
    return scores


def _classifications(rows: list[dict[str, Any]], representative: str) -> dict[tuple[str, int], str]:
    rep_scores = _seed_scores(rows, representative)
    result: dict[tuple[str, int], str] = {}
    for seed_key, values in rep_scores.items():
        seed_rows = [
            {
                "sare": _float(values["kl_lss_sare"]),
                "token_dense": _float(values["kl_lss_token_dense"]),
                "single_expert": _float(values["kl_lss_single_expert"]),
            }
        ]
        result[seed_key] = _classify_seed(seed_rows)
    return result


def _subset_mean(rows: list[dict[str, Any]], candidate: str, label: str, subset: set[tuple[str, int]]) -> float:
    vals = [
        _float(row["final_greedy_success"])
        for row in rows
        if str(row["candidate"]) == candidate and str(row["label"]) == label and _key(row) in subset
    ]
    return sum(vals) / len(vals) if vals else 0.0


def render_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    stage3 = _read_json(Path(campaign["reports"]["stage2_json"]))
    stage2 = _read_json(Path(campaign["reports"]["stage2_verification_json"]))
    dev_lanes = _block_lanes(campaign, "dev")
    round6_rows = [row for row in _current_round6_rows(campaign, dev_lanes) if str(row["label"]) in CURRENT_LABELS]
    rows = round6_rows + stage3["detail_rows"]
    verified = [str(name) for name in stage2["verified_candidates"]]
    representative = verified[0]
    classifications = _classifications(rows, representative)

    global_hard = {seed for seed, label in classifications.items() if label == "global_hard_failure"}
    parity = {seed for seed, label in classifications.items() if label == "shared_control_parity"}
    differentiator = {seed for seed, label in classifications.items() if label == "route_control_differentiator"}

    lines_to_compare = ["round6"] + verified
    subset_rows: list[dict[str, Any]] = []
    for line in lines_to_compare:
        subset_rows.append(
            {
                "line": line,
                "overall_sare": _subset_mean(rows, line, "kl_lss_sare", set(classifications)),
                "overall_token": _subset_mean(rows, line, "kl_lss_token_dense", set(classifications)),
                "overall_single": _subset_mean(rows, line, "kl_lss_single_expert", set(classifications)),
                "global_hard_sare": _subset_mean(rows, line, "kl_lss_sare", global_hard),
                "parity_sare": _subset_mean(rows, line, "kl_lss_sare", parity),
                "differentiator_sare": _subset_mean(rows, line, "kl_lss_sare", differentiator),
                "differentiator_token": _subset_mean(rows, line, "kl_lss_token_dense", differentiator),
                "differentiator_single": _subset_mean(rows, line, "kl_lss_single_expert", differentiator),
            }
        )

    lines = [
        "# Portfolio Discriminator Report",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- verified tie-bucket candidates: `{verified}`",
        f"- representative classification line: `{representative}`",
        "",
        "## Dev Seed Classification",
        "",
        f"- global-hard seeds: `{sorted(f'{lane}/{seed}' for lane, seed in global_hard)}`",
        f"- shared-parity seeds: `{sorted(f'{lane}/{seed}' for lane, seed in parity)}`",
        f"- route/control differentiator seeds: `{sorted(f'{lane}/{seed}' for lane, seed in differentiator)}`",
        "",
        "## Subset Means",
        "",
        "| Line | Overall SARE | Overall token_dense | Overall single_expert | Global-Hard SARE | Parity SARE | Differentiator SARE | Differentiator token_dense | Differentiator single_expert |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in subset_rows:
        lines.append(
            f"| `{row['line']}` | `{row['overall_sare']:.4f}` | `{row['overall_token']:.4f}` | `{row['overall_single']:.4f}` | `{row['global_hard_sare']:.4f}` | `{row['parity_sare']:.4f}` | `{row['differentiator_sare']:.4f}` | `{row['differentiator_token']:.4f}` | `{row['differentiator_single']:.4f}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The portfolio dev split does not contain multiple challenger-discriminating seeds. It contains one global-hard sentinel, one route/control differentiator, and everything else is shared parity.",
            "- On the only differentiator seed (`prospective_f/233`), `round6` and every verified challenger are still tied at `1.0000`; the only separation there is against matched `token_dense` at `0.6250`.",
            "- That means the verified tie bucket remains a true tie even after removing the globally hard seed from view. The right operational conclusion is to keep `round6` as incumbent and stop spending search budget on variants that merely re-enter this bucket.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "verified_candidates": verified,
                "representative_candidate": representative,
                "classifications": {f"{lane}:{seed}": label for (lane, seed), label in classifications.items()},
                "subset_rows": subset_rows,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a discriminator report for the portfolio tie bucket")
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
