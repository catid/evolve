from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _load_yaml, _read_json, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty

SCREENING_SPEC = Path("outputs/reports/portfolio_screening_spec.json")
RESTART_POLICY = Path("outputs/reports/portfolio_restart_policy.json")
CANDIDATE_PACK = Path("outputs/reports/portfolio_candidate_pack.json")


def _tier(candidate: str, restart_default: list[str], reserve_priors: list[str], retired_priors: list[str]) -> str:
    if candidate in restart_default:
        return "restart_default"
    if candidate in reserve_priors:
        return "reserve"
    if candidate in retired_priors:
        return "retired"
    return "unclassified"


def render_pack(campaign: dict[str, Any], output: Path, md_output: Path | None) -> None:
    screening = _read_json(SCREENING_SPEC)
    restart = _read_json(RESTART_POLICY)
    candidate_pack = _read_json(CANDIDATE_PACK)

    restart_default = list(screening["restart_default"])
    reserve_priors = list(screening["reserve_priors"])
    retired_priors = list(screening["retired_priors"])

    seed_roles = dict(screening["seed_roles"])
    candidate_rows = list(screening["candidate_rows"])
    seed_targets: dict[str, dict[str, Any]] = {}
    for role, seed_row in seed_roles.items():
        lane = str(seed_row["lane"])
        seed = int(seed_row["seed"])
        seed_targets[role] = {
            "lane": lane,
            "seed": seed,
        }
    # Enrich with incumbent/control targets from the screening spec markdown inputs.
    # The values are already encoded in the candidate_rows/rules, so keep the pack minimal and stable.
    seed_targets["ranking_support"]["required_min_success"] = 1.0
    seed_targets["ranking_weakness"]["required_min_success"] = float(screening["incumbent_269"]) + 1e-9
    seed_targets["guardrail"]["required_min_success"] = 1.0
    seed_targets["sentinel"]["required_behavior"] = "track_only"

    pack = {
        "schema_version": 1,
        "pack_type": "portfolio_seed_pack",
        "campaign_name": campaign["name"],
        "generated_from": {
            "screening_spec": str(SCREENING_SPEC),
            "restart_policy": str(RESTART_POLICY),
            "active_candidate_pack": str(CANDIDATE_PACK),
        },
        "provenance": {
            "git_commit": get_git_commit(),
            "git_dirty": get_git_dirty(),
        },
        "benchmark": {
            "active_candidate_name": str(candidate_pack["candidate_name"]),
            "active_candidate_pack": str(CANDIDATE_PACK),
            "archived_frozen_pack": str(candidate_pack["frozen_pack_reference"]["path"]),
        },
        "screening_rules": {
            "seed_roles": seed_targets,
            "prune_rules": [
                "prune_support_regression",
                "prune_guardrail_regression",
            ],
            "hold_rules": [
                "tie_only_no_weakness_gain",
                "needs_support_measurement",
            ],
            "advance_rule": "advance_for_broader_dev",
        },
        "frontier": {
            "restart_default": restart_default,
            "reserve_priors": reserve_priors,
            "retired_priors": retired_priors,
            "support_status": restart["support_status"],
            "surviving_candidates": restart["surviving_candidates"],
        },
        "candidate_rows": [
            {
                "candidate": str(row["candidate"]),
                "policy_bucket": str(row["policy_bucket"]),
                "screen_rule": str(row["screen_rule"]),
                "support_233": row["support_233"],
                "weakness_269": row["weakness_269"],
                "guardrail_277": row["guardrail_277"],
                "tier": _tier(str(row["candidate"]), restart_default, reserve_priors, retired_priors),
            }
            for row in candidate_rows
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, pack)

    if md_output is not None:
        lines = [
            "# Portfolio Seed Pack",
            "",
            f"- source campaign: `{campaign['name']}`",
            f"- git commit: `{get_git_commit()}`",
            f"- git dirty: `{get_git_dirty()}`",
            f"- active benchmark: `{pack['benchmark']['active_candidate_name']}`",
            "",
            "## Seed Roles",
            "",
            "| Role | Lane | Seed | Requirement |",
            "| --- | --- | ---: | --- |",
            f"| `sentinel` | `{seed_targets['sentinel']['lane']}` | {seed_targets['sentinel']['seed']} | `{seed_targets['sentinel']['required_behavior']}` |",
            f"| `ranking_support` | `{seed_targets['ranking_support']['lane']}` | {seed_targets['ranking_support']['seed']} | `>= {seed_targets['ranking_support']['required_min_success']:.4f}` |",
            f"| `ranking_weakness` | `{seed_targets['ranking_weakness']['lane']}` | {seed_targets['ranking_weakness']['seed']} | `> {screening['incumbent_269']:.4f}` |",
            f"| `guardrail` | `{seed_targets['guardrail']['lane']}` | {seed_targets['guardrail']['seed']} | `>= {seed_targets['guardrail']['required_min_success']:.4f}` |",
            "",
            "## Frontier Tiers",
            "",
            f"- restart default: `{restart_default}`",
            f"- reserve priors: `{reserve_priors}`",
            f"- retired priors: `{retired_priors}`",
            "",
            "## Candidate Rows",
            "",
            "| Candidate | Tier | Policy Bucket | Screen Rule | support_233 | weakness_269 | guardrail_277 |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
        for row in pack["candidate_rows"]:
            support = "n/a" if row["support_233"] is None else f"{float(row['support_233']):.4f}"
            lines.append(
                f"| `{row['candidate']}` | `{row['tier']}` | `{row['policy_bucket']}` | `{row['screen_rule']}` | `{support}` | `{float(row['weakness_269']):.4f}` | `{float(row['guardrail_277']):.4f}` |"
            )
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- This pack is the operational frontier artifact for future bounded mini-sweeps.",
                "- `round7` is the only restart-default line; `round10` and `round5` are reserve lines; `door3_post5` and `post_unlock_x5` are retired.",
                "- A mini-sweep should consume these seed roles and screen rules before it spends broader DoorKey dev budget.",
            ]
        )
        md_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a machine-readable seed pack from the measured portfolio frontier")
    parser.add_argument("--campaign-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--md-output", required=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign = _load_yaml(Path(args.campaign_config))
    render_pack(campaign, Path(args.output), Path(args.md_output) if args.md_output else None)


if __name__ == "__main__":
    main()
