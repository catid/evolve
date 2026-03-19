from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _read_json, _write_json
from psmn_rl.analysis.portfolio_frontier_contract_loader import load_frontier_contract
from psmn_rl.utils.io import get_git_commit, get_git_dirty

CONSISTENCY_PATH = Path("outputs/reports/portfolio_frontier_consistency.json")


def evaluate_frontier_state(
    active_candidate: str,
    default_restart_prior: str,
    replay_validated_alternate: str,
    consistency_overall: str,
) -> dict[str, Any]:
    checks = [
        {
            "label": "active_candidate_round6",
            "status": "pass" if active_candidate == "round6" else "fail",
            "detail": f"active_candidate={active_candidate}",
        },
        {
            "label": "default_restart_round7",
            "status": "pass" if default_restart_prior == "round7" else "fail",
            "detail": f"default_restart_prior={default_restart_prior}",
        },
        {
            "label": "replay_alternate_round10",
            "status": "pass" if replay_validated_alternate == "round10" else "fail",
            "detail": f"replay_validated_alternate={replay_validated_alternate}",
        },
        {
            "label": "consistency_pass",
            "status": "pass" if consistency_overall == "pass" else "fail",
            "detail": f"consistency_overall={consistency_overall}",
        },
    ]
    overall = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {"overall": overall, "checks": checks}


def render_doctor(output: Path | None, json_output: Path | None) -> dict[str, Any]:
    contract = load_frontier_contract()
    consistency = _read_json(CONSISTENCY_PATH)
    result = evaluate_frontier_state(
        active_candidate=contract.benchmark.active_candidate,
        default_restart_prior=contract.frontier_roles.default_restart_prior,
        replay_validated_alternate=contract.frontier_roles.replay_validated_alternate,
        consistency_overall=str(consistency["overall"]),
    )

    if output is not None:
        lines = [
            "# Portfolio Frontier Doctor",
            "",
            f"- git commit: `{get_git_commit()}`",
            f"- git dirty: `{get_git_dirty()}`",
            f"- overall: `{result['overall']}`",
            "",
            "## Checks",
            "",
            "| Check | Status | Detail |",
            "| --- | --- | --- |",
        ]
        for check in result["checks"]:
            lines.append(f"| `{check['label']}` | `{check['status']}` | {check['detail']} |")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if json_output is not None:
        _write_json(json_output, result)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Doctor check for the frozen portfolio frontier state")
    parser.add_argument("--output", required=False)
    parser.add_argument("--json", required=False)
    parser.add_argument("--fail-on-drift", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = render_doctor(
        Path(args.output) if args.output else None,
        Path(args.json) if args.json else None,
    )
    if args.fail_on_drift and result["overall"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
