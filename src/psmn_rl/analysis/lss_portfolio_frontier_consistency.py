from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _load_yaml, _read_json, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty

RESTART_POLICY = Path("outputs/reports/portfolio_restart_policy.json")
SEED_PACK = Path("outputs/reports/portfolio_seed_pack.json")
SEED_PACK_SCORER = Path("outputs/reports/portfolio_seed_pack_scorer.json")
FRONTIER_REPLAY = Path("outputs/reports/portfolio_frontier_replay.json")
FRONTIER_MANIFEST = Path("outputs/reports/portfolio_frontier_manifest.json")


def _check(label: str, condition: bool, detail: str) -> dict[str, Any]:
    return {
        "label": label,
        "status": "pass" if condition else "fail",
        "detail": detail,
    }


def render_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    restart = _read_json(RESTART_POLICY)
    seed_pack = _read_json(SEED_PACK)
    scorer = _read_json(SEED_PACK_SCORER)
    replay = _read_json(FRONTIER_REPLAY)
    manifest = _read_json(FRONTIER_MANIFEST)

    scorer_map = {str(row["candidate"]): str(row["verdict"]) for row in scorer["rows"]}
    replay_map = {str(row["case_candidate"]): str(row["verdict"]) for row in replay["rows"]}
    manifest_map = {str(row["candidate"]): str(row["manifest_bucket"]) for row in manifest["rows"]}

    checks = [
        _check(
            "active_benchmark_round6",
            manifest["active"] == ["round6"],
            f"manifest active={manifest['active']}",
        ),
        _check(
            "restart_default_round7",
            restart["restart_default"] == ["round7"]
            and manifest["default_restart"] == ["round7"]
            and seed_pack["frontier"]["restart_default"] == ["round7"],
            f"restart={restart['restart_default']} manifest={manifest['default_restart']} pack={seed_pack['frontier']['restart_default']}",
        ),
        _check(
            "retired_priors_match",
            restart["retired_priors"] == manifest["retired"] == seed_pack["frontier"]["retired_priors"],
            f"restart={restart['retired_priors']} manifest={manifest['retired']} pack={seed_pack['frontier']['retired_priors']}",
        ),
        _check(
            "round7_scorer_advances",
            scorer_map.get("round7") == "advance_for_broader_dev",
            f"scorer round7={scorer_map.get('round7')}",
        ),
        _check(
            "round10_replay_advances",
            replay_map.get("round10") == "advance_for_broader_dev"
            and manifest["replay_validated_alternates"] == ["round10"],
            f"replay round10={replay_map.get('round10')} manifest={manifest['replay_validated_alternates']}",
        ),
        _check(
            "round5_hold_only",
            scorer_map.get("round5") == "hold_seed_clean_but_below_incumbent"
            and manifest["hold_only"] == ["round5"],
            f"scorer round5={scorer_map.get('round5')} manifest={manifest['hold_only']}",
        ),
        _check(
            "door3_retired_consistent",
            scorer_map.get("door3_post5") == "prune_support_regression"
            and manifest_map.get("door3_post5") == "retired_prior",
            f"scorer door3={scorer_map.get('door3_post5')} manifest={manifest_map.get('door3_post5')}",
        ),
        _check(
            "post_unlock_x5_retired_consistent",
            scorer_map.get("post_unlock_x5") == "prune_guardrail_regression"
            and manifest_map.get("post_unlock_x5") == "retired_prior",
            f"scorer post_unlock_x5={scorer_map.get('post_unlock_x5')} manifest={manifest_map.get('post_unlock_x5')}",
        ),
    ]

    overall = "pass" if all(check["status"] == "pass" for check in checks) else "fail"

    lines = [
        "# Portfolio Frontier Consistency Check",
        "",
        f"- source campaign: `{campaign['name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- overall: `{overall}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in checks:
        lines.append(f"| `{check['label']}` | `{check['status']}` | {check['detail']} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is the drift check for the measured frontier stack.",
            "- It should stay green as long as the restart policy, seed pack, scorer, replay, and manifest all encode the same operational state.",
            "- Any future bounded search update that changes one frontier artifact without the others should trip this report before the repo starts giving contradictory guidance.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_output is not None:
        _write_json(json_output, {"overall": overall, "checks": checks})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check consistency across the measured portfolio frontier artifacts")
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
