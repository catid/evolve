from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.portfolio_frontier_contract_loader import load_frontier_contract
from psmn_rl.analysis.lss_post_pass_campaign import _load_yaml, _read_json, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty

SCHEDULE = Path("outputs/reports/portfolio_frontier_schedule.json")


def render_kit(campaign: dict[str, Any], output: Path, md_output: Path | None) -> None:
    contract = load_frontier_contract()
    schedule = _read_json(SCHEDULE)

    queue = list(schedule["rows"])
    first_restart = next((row for row in queue if row["bucket"] == "default_restart"), None)
    second_restart = next((row for row in queue if row["bucket"] == "validated_alternate"), None)

    kit = {
        "schema_version": 1,
        "kit_type": "portfolio_frontier_kit",
        "campaign_name": campaign["name"],
        "provenance": {
            "git_commit": get_git_commit(),
            "git_dirty": get_git_dirty(),
        },
        "benchmark": {
            "active_candidate": contract.benchmark.active_candidate,
            "active_candidate_pack": contract.benchmark.active_candidate_pack,
            "archived_frozen_pack": contract.benchmark.archived_frozen_pack,
        },
        "ordered_queue": queue,
        "next_restart": {
            "primary": first_restart["candidate"] if first_restart is not None else None,
            "secondary": second_restart["candidate"] if second_restart is not None else None,
        },
        "seed_contract": {
            "sentinel": {
                "lane": contract.role_seed("sentinel").lane,
                "seed": contract.role_seed("sentinel").seed,
                "mode": "track_only",
            },
            "ranking_support": {
                "lane": contract.role_seed("ranking_support").lane,
                "seed": contract.role_seed("ranking_support").seed,
                "required_min_success": contract.support_min_success,
            },
            "ranking_weakness": {
                "lane": contract.role_seed("ranking_weakness").lane,
                "seed": contract.role_seed("ranking_weakness").seed,
                "required_strictly_above": contract.weakness_min_success_exclusive,
            },
            "guardrail": {
                "lane": contract.role_seed("guardrail").lane,
                "seed": contract.role_seed("guardrail").seed,
                "required_min_success": contract.guardrail_min_success,
            },
        },
        "promotion_rules": {
            "prune_on_support_regression": True,
            "prune_on_guardrail_regression": True,
            "hold_on_weakness_tie": True,
            "broader_dev_only_after_seed_clear": True,
        },
        "frontier_roles": {
            "default_restart_prior": contract.frontier_roles.default_restart_prior,
            "replay_validated_alternate": contract.frontier_roles.replay_validated_alternate,
            "hold_only_priors": list(contract.frontier_roles.hold_only_priors),
            "retired_priors": list(contract.frontier_roles.retired_priors),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, kit)

    if md_output is not None:
        lines = [
            "# Portfolio Frontier Kit",
            "",
            f"- source campaign: `{campaign['name']}`",
            f"- git commit: `{get_git_commit()}`",
            f"- git dirty: `{get_git_dirty()}`",
            f"- active benchmark: `{kit['benchmark']['active_candidate']}`",
            "",
            "## Next Restart",
            "",
            f"- primary: `{kit['next_restart']['primary']}`",
            f"- secondary: `{kit['next_restart']['secondary']}`",
            "",
            "## Queue",
            "",
            "| Priority | Candidate | Bucket | Action |",
            "| ---: | --- | --- | --- |",
        ]
        for row in queue:
            lines.append(
                f"| `{row['priority']}` | `{row['candidate']}` | `{row['bucket']}` | `{row['action']}` |"
            )
        lines.extend(
            [
                "",
                "## Seed Contract",
                "",
                f"- sentinel: `{kit['seed_contract']['sentinel']}`",
                f"- ranking_support: `{kit['seed_contract']['ranking_support']}`",
                f"- ranking_weakness: `{kit['seed_contract']['ranking_weakness']}`",
                f"- guardrail: `{kit['seed_contract']['guardrail']}`",
                "",
                "## Promotion Rules",
                "",
                f"- rules: `{kit['promotion_rules']}`",
                "",
                "## Interpretation",
                "",
                "- This kit is the consumable handoff artifact for the next bounded restart.",
                "- It tells future search exactly which prior to run first, which alternate to try next, which lines to keep on hold, and which ones are retired.",
                "- It also carries the measured seed contract and promotion rules in one place so downstream tools do not need to reconstruct them from multiple reports.",
            ]
        )
        md_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a consumable runbook kit from the frozen portfolio frontier")
    parser.add_argument("--campaign-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--md-output", required=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign = _load_yaml(Path(args.campaign_config))
    render_kit(campaign, Path(args.output), Path(args.md_output) if args.md_output else None)


if __name__ == "__main__":
    main()
