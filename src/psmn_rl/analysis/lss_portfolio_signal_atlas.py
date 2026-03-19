from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _float, _load_yaml, _read_json, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty

LABEL_DIRS = {
    "sare": "kl_lss_sare",
    "token_dense": "kl_lss_token_dense",
    "single_expert": "kl_lss_single_expert",
}


def _summary(root: Path, seed: int, variant: str) -> dict[str, Any]:
    return _read_json(root / f"seed_{seed}" / LABEL_DIRS[variant] / "summary.json")


def _classify_seed(sare: float, token: float, single: float, eps: float = 1e-9) -> str:
    if sare <= eps and token <= eps and single <= eps:
        return "global_hard_failure"
    if sare + eps < max(token, single):
        return "control_advantaged"
    if sare > token + eps or sare > single + eps:
        return "benchmark_support"
    return "shared_control_parity"


def _row(campaign: dict[str, Any], lane: str, seed: int, group: str) -> dict[str, Any]:
    sare_root = Path(campaign["current_round6_sare_roots"][lane])
    ctrl_root = Path(campaign["current_round6_control_roots"][lane])
    sare = _float(_summary(sare_root, seed, "sare")["final_greedy_success"])
    token = _float(_summary(ctrl_root, seed, "token_dense")["final_greedy_success"])
    single = _float(_summary(ctrl_root, seed, "single_expert")["final_greedy_success"])
    return {
        "group": group,
        "lane": lane,
        "seed": seed,
        "sare": sare,
        "token_dense": token,
        "single_expert": single,
        "sare_minus_token": sare - token,
        "sare_minus_single": sare - single,
        "classification": _classify_seed(sare, token, single),
    }


def render_atlas(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    rows: list[dict[str, Any]] = []
    for group in ("dev", "holdout", "healthy"):
        for block in campaign["blocks"][group]:
            lane = str(block["lane"])
            for seed in block["seeds"]:
                rows.append(_row(campaign, lane, int(seed), group))

    counts: dict[str, int] = {}
    group_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        label = str(row["classification"])
        counts[label] = counts.get(label, 0) + 1
        group_counts.setdefault(str(row["group"]), {})
        group_counts[str(row["group"])][label] = group_counts[str(row["group"])].get(label, 0) + 1

    informative = [row for row in rows if row["classification"] in {"benchmark_support", "control_advantaged"}]

    lines = [
        "# Portfolio Signal Atlas",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- dev families: `{[block['lane'] for block in campaign['blocks']['dev']]}`",
        f"- holdout families: `{[block['lane'] for block in campaign['blocks']['holdout']]}`",
        f"- healthy families: `{[block['lane'] for block in campaign['blocks']['healthy']]}`",
        "",
        "## Classification Counts",
        "",
        f"- overall counts: `{counts}`",
        f"- counts by group: `{group_counts}`",
        "",
        "## Seed Atlas",
        "",
        "| Group | Lane | Seed | SARE | token_dense | single_expert | SARE-token | SARE-single | Classification |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['group']}` | `{row['lane']}` | {row['seed']} | `{row['sare']:.4f}` | `{row['token_dense']:.4f}` | `{row['single_expert']:.4f}` | `{row['sare_minus_token']:.4f}` | `{row['sare_minus_single']:.4f}` | `{row['classification']}` |"
        )

    lines.extend(
        [
            "",
            "## Informative Seeds Only",
            "",
            "| Group | Lane | Seed | Classification | SARE | token_dense | single_expert |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in informative:
        lines.append(
            f"| `{row['group']}` | `{row['lane']}` | {row['seed']} | `{row['classification']}` | `{row['sare']:.4f}` | `{row['token_dense']:.4f}` | `{row['single_expert']:.4f}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `global_hard_failure` seeds are poor challenger discriminators because every compared line is failing together.",
            "- `shared_control_parity` seeds tell you the family is competent there, but they do not separate tied challengers from the incumbent.",
            "- `benchmark_support` seeds are the useful support set for the active benchmark: they show where `round6` is at least as good as the controls and materially above at least one of them.",
            "- `control_advantaged` seeds are the bounded weakness set for future work: they show where matched controls still retain an advantage and therefore deserve targeted mechanism work if the project reopens challenger search.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "rows": rows,
                "counts": counts,
                "group_counts": group_counts,
                "informative_rows": informative,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an active-benchmark signal atlas from the portfolio roots")
    parser.add_argument("--campaign-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", required=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign = _load_yaml(Path(args.campaign_config))
    render_atlas(campaign, Path(args.output), Path(args.json) if args.json else None)


if __name__ == "__main__":
    main()
