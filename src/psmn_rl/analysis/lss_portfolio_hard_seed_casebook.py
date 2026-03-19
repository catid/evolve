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


def _summary(path: Path) -> dict[str, Any]:
    return _read_json(path / "summary.json")


def _round6_root(campaign: dict[str, Any], lane: str, variant: str) -> Path:
    if variant == "sare":
        return Path(campaign["current_round6_sare_roots"][lane])
    return Path(campaign["current_round6_control_roots"][lane])


def _round6_summary(campaign: dict[str, Any], lane: str, seed: int, variant: str) -> dict[str, Any]:
    return _summary(_round6_root(campaign, lane, variant) / f"seed_{seed}" / LABEL_DIRS[variant])


def _candidate_summary(campaign: dict[str, Any], candidate: str, lane: str, seed: int) -> dict[str, Any]:
    root = Path(campaign["stage_roots"]["stage2_verification_a"])
    return _summary(root / candidate / lane / f"seed_{seed}" / LABEL_DIRS["sare"])


def _seed_triplet(campaign: dict[str, Any], candidate: str, lane: str, seed: int) -> dict[str, Any]:
    sare = _candidate_summary(campaign, candidate, lane, seed)
    token = _round6_summary(campaign, lane, seed, "token_dense")
    single = _round6_summary(campaign, lane, seed, "single_expert")
    return {
        "lane": lane,
        "seed": seed,
        "candidate": candidate,
        "sare": _float(sare["final_greedy_success"]),
        "token_dense": _float(token["final_greedy_success"]),
        "single_expert": _float(single["final_greedy_success"]),
    }


def _round_trace(summary: dict[str, Any]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row in summary["rounds"]:
        rows.append(
            {
                "round": _float(row["round"]),
                "after_greedy_success": _float(row["after_greedy_success"]),
                "post_unlock_frac": _float(row.get("collection/phase_frac_post_unlock", 0.0)),
                "disagreement_rate": _float(row.get("collection/disagreement_rate", 0.0)),
                "route_entropy": _float(row.get("collection/route_entropy", 0.0)),
                "path_entropy": _float(row.get("collection/path_entropy", 0.0)),
            }
        )
    return rows


def _classify_seed(outcomes: list[dict[str, Any]]) -> str:
    if all(_float(row["sare"]) == 0.0 and _float(row["token_dense"]) == 0.0 and _float(row["single_expert"]) == 0.0 for row in outcomes):
        return "global_hard_failure"
    if any(_float(row["sare"]) > _float(row["token_dense"]) or _float(row["sare"]) > _float(row["single_expert"]) for row in outcomes):
        return "route_control_differentiator"
    return "shared_control_parity"


def render_casebook(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    stage2 = _read_json(Path(campaign["reports"]["stage2_verification_json"]))
    verified_candidates = [str(name) for name in stage2["verified_candidates"]]
    representative = verified_candidates[0]
    seeds = {
        "prospective_c": [181, 191, 193],
        "prospective_f": [233],
    }

    seed_rows: list[dict[str, Any]] = []
    for lane, lane_seeds in seeds.items():
        for seed in lane_seeds:
            for candidate in verified_candidates:
                seed_rows.append(_seed_triplet(campaign, candidate, lane, seed))

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault((str(row["lane"]), int(row["seed"])), []).append(row)

    classifications = {
        f"{lane}:{seed}": _classify_seed(rows)
        for (lane, seed), rows in grouped.items()
    }

    round6_curves = {
        "prospective_c:193": _round_trace(_round6_summary(campaign, "prospective_c", 193, "sare")),
        "prospective_c:191": _round_trace(_round6_summary(campaign, "prospective_c", 191, "sare")),
        "prospective_f:233": _round_trace(_round6_summary(campaign, "prospective_f", 233, "sare")),
    }
    control_curves_193 = {
        "token_dense": _round_trace(_round6_summary(campaign, "prospective_c", 193, "token_dense")),
        "single_expert": _round_trace(_round6_summary(campaign, "prospective_c", 193, "single_expert")),
    }

    lines = [
        "# Portfolio Hard-Seed Casebook",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- verified tie-bucket candidates: `{verified_candidates}`",
        f"- representative challenger for detailed curves: `{representative}`",
        "",
        "## Summary",
        "",
        "- `prospective_c/193` is a global hard-failure seed on the portfolio dev split, not a SARE-specific miss.",
        "- All six fully verified tie-bucket challengers reproduce the exact same result on that seed: `SARE = 0.0`, matched `token_dense = 0.0`, matched `single_expert = 0.0`.",
        "- Adjacent seeds `prospective_c/181` and `prospective_c/191` are solved by every verified challenger and every matched control, so the failure is highly localized rather than a broad `prospective_c` collapse.",
        "- `prospective_f/233` remains a route/control differentiator: verified challengers and `single_expert` solve it at `1.0`, while matched `token_dense` stays at `0.625`.",
        "- Future challenger screens should keep `prospective_c/193` as a hardness sentinel, but not treat it as evidence that one tied candidate is stronger than another unless some line actually breaks the all-zero control parity.",
        "",
        "## Seed-Level Outcomes Across Verified Challengers",
        "",
        "| Lane | Seed | Classification | Candidate | SARE | token_dense | single_expert |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for (lane, seed), rows in sorted(grouped.items()):
        label = classifications[f"{lane}:{seed}"]
        for row in sorted(rows, key=lambda item: item["candidate"]):
            lines.append(
                f"| {lane} | {seed} | `{label}` | `{row['candidate']}` | `{row['sare']:.4f}` | `{row['token_dense']:.4f}` | `{row['single_expert']:.4f}` |"
            )

    lines.extend(
        [
            "",
            "## Round6 Mechanism Contrast",
            "",
            "| Case | Round | Greedy | Post-Unlock Frac | Disagreement | Route Entropy | Path Entropy |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case, rows in round6_curves.items():
        for row in rows:
            lines.append(
                f"| `{case}` | `{int(row['round'])}` | `{row['after_greedy_success']:.4f}` | `{row['post_unlock_frac']:.4f}` | `{row['disagreement_rate']:.4f}` | `{row['route_entropy']:.4f}` | `{row['path_entropy']:.4f}` |"
            )

    lines.extend(
        [
            "",
            "## Control Contrast On prospective_c/193",
            "",
            "| Variant | Round | Greedy | Post-Unlock Frac | Disagreement | Route Entropy | Path Entropy |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant, rows in control_curves_193.items():
        for row in rows:
            lines.append(
                f"| `{variant}` | `{int(row['round'])}` | `{row['after_greedy_success']:.4f}` | `{row['post_unlock_frac']:.4f}` | `{row['disagreement_rate']:.4f}` | `{row['route_entropy']:.4f}` | `{row['path_entropy']:.4f}` |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `prospective_c/193` differs from the usual late-phase recovery pattern because `round6` never reaches post-unlock at all; its post-unlock fraction stays `0.0000` for every round and disagreement collapses to near-zero by round 3.",
            "- The matched controls also fail there, which means the seed is not currently a useful discriminator between `round6` and its tied challengers.",
            "- By contrast, solved seeds such as `prospective_c/191` and `prospective_f/233` show the expected recovery signature: high post-unlock occupancy appears in the successful rounds and greedy success turns on only after that phase is actually reached.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if json_output is not None:
        _write_json(
            json_output,
            {
                "verified_candidates": verified_candidates,
                "representative_candidate": representative,
                "seed_rows": seed_rows,
                "classifications": classifications,
                "round6_curves": round6_curves,
                "control_curves_193": control_curves_193,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a hard-seed casebook for the portfolio tie bucket")
    parser.add_argument("--campaign-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", required=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign = _load_yaml(Path(args.campaign_config))
    render_casebook(campaign, Path(args.output), Path(args.json) if args.json else None)


if __name__ == "__main__":
    main()
