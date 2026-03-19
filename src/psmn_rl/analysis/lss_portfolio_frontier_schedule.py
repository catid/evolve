from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.portfolio_frontier_contract_loader import load_frontier_contract
from psmn_rl.analysis.lss_post_pass_campaign import _load_yaml, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty


def _priority(bucket: str) -> int:
    order = {
        "default_restart": 1,
        "validated_alternate": 2,
        "hold_only": 3,
        "retired": 4,
    }
    return order[bucket]


def render_schedule(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    contract = load_frontier_contract()

    rows = [
        {
            "candidate": contract.frontier_roles.default_restart_prior,
            "bucket": "default_restart",
            "priority": _priority("default_restart"),
            "action": "run_first",
            "reason": "lowest-friction measured restart prior",
        },
        {
            "candidate": contract.frontier_roles.replay_validated_alternate,
            "bucket": "validated_alternate",
            "priority": _priority("validated_alternate"),
            "action": "run_second_if_needed",
            "reason": "only replay-validated alternate on the historically stitched surface",
        },
    ]

    rows.extend(
        {
            "candidate": candidate,
            "bucket": "hold_only",
            "priority": _priority("hold_only"),
            "action": "defer_until_restart_and_alternate_fail",
            "reason": "seed-clean but below incumbent on broader dev mean",
        }
        for candidate in contract.frontier_roles.hold_only_priors
    )
    rows.extend(
        {
            "candidate": candidate,
            "bucket": "retired",
            "priority": _priority("retired"),
            "action": "do_not_restart",
            "reason": "measured support or guardrail regression",
        }
        for candidate in contract.frontier_roles.retired_priors
    )
    rows.sort(key=lambda row: (int(row["priority"]), str(row["candidate"])))

    seed_contract = {
        "sentinel": {
            "lane": contract.role_seed("sentinel").lane,
            "seed": contract.role_seed("sentinel").seed,
            "use": "track_only",
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
    }

    lines = [
        "# Portfolio Frontier Schedule",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- active benchmark: `{contract.benchmark.active_candidate}`",
        "",
        "## Ordered Queue",
        "",
        "| Priority | Candidate | Bucket | Action | Reason |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['priority']}` | `{row['candidate']}` | `{row['bucket']}` | `{row['action']}` | {row['reason']} |"
        )

    lines.extend(
        [
            "",
            "## Measured Seed Contract",
            "",
            f"- sentinel: `{seed_contract['sentinel']}`",
            f"- ranking_support: `{seed_contract['ranking_support']}`",
            f"- ranking_weakness: `{seed_contract['ranking_weakness']}`",
            f"- guardrail: `{seed_contract['guardrail']}`",
            "",
            "## Stop Rules",
            "",
            "- prune immediately on support regression below `1.0000` at `prospective_f/233`",
            "- prune immediately on guardrail regression below `1.0000` at `prospective_h/277`",
            "- do not advance lines that only tie the incumbent on `prospective_h/269`",
            "- only spend broader dev/fairness budget after clearing the measured seed contract",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(
            json_output,
            {
                "active_benchmark": contract.benchmark.active_candidate,
                "rows": rows,
                "seed_contract": seed_contract,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an ordered execution schedule from the frozen portfolio frontier")
    parser.add_argument("--campaign-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", required=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign = _load_yaml(Path(args.campaign_config))
    render_schedule(campaign, Path(args.output), Path(args.json) if args.json else None)


if __name__ == "__main__":
    main()
