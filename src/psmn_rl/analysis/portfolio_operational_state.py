from __future__ import annotations

import argparse
from pathlib import Path

from psmn_rl.analysis.lss_post_pass_campaign import _write_json
from psmn_rl.analysis.portfolio_active_state_doctor_loader import load_active_state_doctor_report
from psmn_rl.analysis.portfolio_candidate_pack_loader import load_portfolio_candidate_pack
from psmn_rl.analysis.portfolio_frontier_contract_loader import load_frontier_contract
from psmn_rl.analysis.portfolio_frontier_guard_report_loader import load_frontier_guard_report
from psmn_rl.analysis.portfolio_gate_report_loader import load_portfolio_gate_report
from psmn_rl.utils.io import get_git_commit, get_git_dirty


DECISION_MEMO_PATH = "outputs/reports/portfolio_decision_memo.md"
GATE_REPORT_PATH = "outputs/reports/portfolio_gate_report.md"


def build_operational_state() -> dict[str, object]:
    candidate_pack = load_portfolio_candidate_pack()
    gate_report = load_portfolio_gate_report()
    contract = load_frontier_contract()
    guard_report = load_frontier_guard_report()
    active_state_doctor = load_active_state_doctor_report()
    return {
        "status_type": "portfolio_operational_state",
        "active_candidate": candidate_pack.candidate_name,
        "active_candidate_pack": candidate_pack.portfolio_campaign.active_canonical_pack,
        "archived_frozen_pack": candidate_pack.portfolio_campaign.archived_legacy_pack,
        "gate_verdict": gate_report.verdict,
        "gate_mode": gate_report.mode,
        "active_pack_role": candidate_pack.active_benchmark_state.active_pack_role,
        "default_restart_prior": contract.frontier_roles.default_restart_prior,
        "replay_validated_alternate": contract.frontier_roles.replay_validated_alternate,
        "evaluation_task": candidate_pack.evaluation.task,
        "evaluation_path_key": candidate_pack.evaluation.path_key,
        "evaluation_episodes": candidate_pack.evaluation.episodes,
        "frontier_guard_overall": guard_report.overall,
        "active_state_doctor_overall": active_state_doctor.overall,
        "ready_for_bounded_restart": (
            guard_report.overall == "pass"
            and active_state_doctor.overall == "pass"
            and gate_report.verdict == "PASS: thaw consideration allowed"
        ),
        "decision_reference": DECISION_MEMO_PATH,
        "gate_report_reference": GATE_REPORT_PATH,
    }


def render_operational_state(output: Path | None, json_output: Path | None) -> dict[str, object]:
    result = build_operational_state()

    if output is not None:
        lines = [
            "# Portfolio Operational State",
            "",
            f"- git commit: `{get_git_commit()}`",
            f"- git dirty: `{get_git_dirty()}`",
            f"- active candidate: `{result['active_candidate']}`",
            f"- active candidate pack: `{result['active_candidate_pack']}`",
            f"- archived frozen pack: `{result['archived_frozen_pack']}`",
            f"- gate verdict: `{result['gate_verdict']}`",
            f"- default restart prior: `{result['default_restart_prior']}`",
            f"- replay-validated alternate: `{result['replay_validated_alternate']}`",
            f"- frontier guard overall: `{result['frontier_guard_overall']}`",
            f"- active-state doctor overall: `{result['active_state_doctor_overall']}`",
            f"- ready for bounded restart: `{result['ready_for_bounded_restart']}`",
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if json_output is not None:
        _write_json(json_output, result)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the machine-readable portfolio operational state snapshot")
    parser.add_argument("--output", required=False)
    parser.add_argument("--json", required=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    render_operational_state(
        Path(args.output) if args.output else None,
        Path(args.json) if args.json else None,
    )


if __name__ == "__main__":
    main()
