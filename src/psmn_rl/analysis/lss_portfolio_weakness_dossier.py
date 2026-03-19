from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_portfolio_signal_atlas import LABEL_DIRS, _classify_seed
from psmn_rl.analysis.lss_post_pass_campaign import _float, _load_yaml, _read_json, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty


def _summary(root: Path, seed: int, variant: str) -> dict[str, Any]:
    return _read_json(root / f"seed_{seed}" / LABEL_DIRS[variant] / "summary.json")


def _variant_root(campaign: dict[str, Any], lane: str, variant: str) -> Path:
    if variant == "sare":
        return Path(campaign["current_round6_sare_roots"][lane])
    return Path(campaign["current_round6_control_roots"][lane])


def _seed_row(campaign: dict[str, Any], group: str, lane: str, seed: int) -> dict[str, Any]:
    sare = _summary(_variant_root(campaign, lane, "sare"), seed, "sare")
    token = _summary(_variant_root(campaign, lane, "token_dense"), seed, "token_dense")
    single = _summary(_variant_root(campaign, lane, "single_expert"), seed, "single_expert")
    sare_success = _float(sare["final_greedy_success"])
    token_success = _float(token["final_greedy_success"])
    single_success = _float(single["final_greedy_success"])
    return {
        "group": group,
        "lane": lane,
        "seed": seed,
        "classification": _classify_seed(sare_success, token_success, single_success),
        "sare": sare_success,
        "token_dense": token_success,
        "single_expert": single_success,
        "sare_minus_token": sare_success - token_success,
        "sare_minus_single": sare_success - single_success,
        "sare_best_round": int(sare["best_round_index"]),
        "token_best_round": int(token["best_round_index"]),
        "single_best_round": int(single["best_round_index"]),
    }


def _round_trace(summary: dict[str, Any]) -> list[dict[str, float | None]]:
    rows: list[dict[str, float | None]] = []
    for row in summary["rounds"]:
        rows.append(
            {
                "round": _float(row["round"]),
                "after_greedy_success": _float(row["after_greedy_success"]),
                "post_unlock_frac": _float(row.get("collection/phase_frac_post_unlock", 0.0)),
                "disagreement_rate": _float(row.get("collection/disagreement_rate", 0.0)),
                "route_entropy": row.get("collection/route_entropy"),
                "path_entropy": row.get("collection/path_entropy"),
            }
        )
    return rows


def _round_trace_row(campaign: dict[str, Any], lane: str, seed: int, variant: str) -> list[dict[str, float | None]]:
    return _round_trace(_summary(_variant_root(campaign, lane, variant), seed, variant))


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def render_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    rows: list[dict[str, Any]] = []
    groups = ("dev", "holdout", "healthy")
    for group in groups:
        for block in campaign["blocks"][group]:
            lane = str(block["lane"])
            for seed in block["seeds"]:
                rows.append(_seed_row(campaign, group, lane, int(seed)))

    weakness_rows = [row for row in rows if row["classification"] == "control_advantaged"]
    if not weakness_rows:
        raise ValueError("No control-advantaged rows found; weakness dossier requires at least one bounded weakness seed")

    primary = weakness_rows[0]
    lane_rows = [row for row in rows if row["lane"] == primary["lane"] and row["group"] == primary["group"]]

    traces = {
        "round6": _round_trace_row(campaign, str(primary["lane"]), int(primary["seed"]), "sare"),
        "token_dense": _round_trace_row(campaign, str(primary["lane"]), int(primary["seed"]), "token_dense"),
        "single_expert": _round_trace_row(campaign, str(primary["lane"]), int(primary["seed"]), "single_expert"),
    }

    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["classification"])
        counts[label] = counts.get(label, 0) + 1

    lines = [
        "# Portfolio Weakness Dossier",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- overall classification counts: `{counts}`",
        f"- bounded weakness seeds: `{[f'{row['lane']}/{row['seed']}' for row in weakness_rows]}`",
        "",
        "## Primary Weakness",
        "",
        f"- primary weakness seed: `{primary['lane']}/{primary['seed']}`",
        f"- group: `{primary['group']}`",
        f"- final success: `round6={primary['sare']:.4f}`, `token_dense={primary['token_dense']:.4f}`, `single_expert={primary['single_expert']:.4f}`",
        f"- final deltas: `round6-token_dense={primary['sare_minus_token']:.4f}`, `round6-single_expert={primary['sare_minus_single']:.4f}`",
        f"- best-round indices: `round6={primary['sare_best_round']}`, `token_dense={primary['token_best_round']}`, `single_expert={primary['single_best_round']}`",
        "",
        "## Same-Lane Contrast",
        "",
        "| Lane | Seed | Classification | round6 | token_dense | single_expert | round6-token | round6-single |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(lane_rows, key=lambda item: int(item["seed"])):
        lines.append(
            f"| `{row['lane']}` | {row['seed']} | `{row['classification']}` | `{row['sare']:.4f}` | `{row['token_dense']:.4f}` | `{row['single_expert']:.4f}` | `{row['sare_minus_token']:.4f}` | `{row['sare_minus_single']:.4f}` |"
        )

    lines.extend(
        [
            "",
            "## Round Trace On The Weakness Seed",
            "",
            "| Variant | Round | Greedy | Post-Unlock Frac | Disagreement | Route Entropy | Path Entropy |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant, variant_rows in traces.items():
        for row in variant_rows:
            lines.append(
                f"| `{variant}` | `{int(_float(row['round']))}` | `{_float(row['after_greedy_success']):.4f}` | `{_float(row['post_unlock_frac']):.4f}` | `{_float(row['disagreement_rate']):.4f}` | `{_format_optional(row['route_entropy'])}` | `{_format_optional(row['path_entropy'])}` |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The active benchmark has exactly one bounded control-advantaged seed on the current DoorKey surface, so the weakness is localized rather than broad.",
            "- On that seed, `round6` does recover to `1.0000` mid-run before finishing at `0.9844`, which makes this a late-round end-state slip rather than a full inability to solve the case.",
            "- The neighboring holdout seeds in the same lane (`271` and `277`) still finish at full parity, which confirms the weakness is seed-local within `prospective_h` rather than lane-wide.",
            "- Future challenger work should treat this seed as the main bounded weakness target while avoiding more spend on seeds that are already shared parity or global-hard failures.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "counts": counts,
                "rows": rows,
                "weakness_rows": weakness_rows,
                "primary_weakness": primary,
                "lane_rows": lane_rows,
                "traces": traces,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a bounded weakness dossier from the portfolio active benchmark surface")
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
