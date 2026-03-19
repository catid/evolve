from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _write_json
from psmn_rl.analysis.portfolio_frontier_contract_loader import load_frontier_contract
from psmn_rl.analysis.portfolio_seed_pack_loader import load_portfolio_seed_pack
from psmn_rl.analysis.portfolio_seed_pack_scorer_loader import SeedPackScorerReport, load_seed_pack_scorer_report
from psmn_rl.analysis.portfolio_seed_pack_validation_loader import (
    SeedPackValidationReport,
    load_seed_pack_validation_report,
)
from psmn_rl.utils.io import get_git_commit, get_git_dirty


SEED_PACK_PATH = Path("outputs/reports/portfolio_seed_pack.json")
SEED_PACK_VALIDATION_PATH = Path("outputs/reports/portfolio_seed_pack_validation.json")
SEED_PACK_SCORER_PATH = Path("outputs/reports/portfolio_seed_pack_scorer.json")
EXPECTED_ACTIVE_PACK = "outputs/reports/portfolio_candidate_pack.json"
EXPECTED_ARCHIVED_PACK = "outputs/reports/frozen_benchmark_pack.json"

def _candidate_set(values: tuple[str, ...]) -> set[str]:
    return set(values)


def evaluate_seed_pack_state(
    validation: SeedPackValidationReport,
    scorer: SeedPackScorerReport,
) -> dict[str, Any]:
    seed_pack = load_portfolio_seed_pack(SEED_PACK_PATH)
    contract = load_frontier_contract()

    validation_restart_default = _candidate_set(validation.validated_restart_default)
    validation_reserve = _candidate_set(validation.validated_reserve)
    validation_retired = _candidate_set(validation.validated_retired)
    validation_bucket_by_candidate = {
        row.candidate: row.validation_bucket for row in validation.rows
    }
    scorer_verdict_by_candidate = {row.candidate: row.verdict for row in scorer.rows}

    expected_reserve = {contract.frontier_roles.replay_validated_alternate, *contract.frontier_roles.hold_only_priors}
    seed_roles_match = (
        seed_pack.screening_rules.seed_roles["ranking_support"].lane == contract.role_seed("ranking_support").lane
        and seed_pack.screening_rules.seed_roles["ranking_support"].seed == contract.role_seed("ranking_support").seed
        and seed_pack.screening_rules.seed_roles["ranking_support"].required_min_success == contract.support_min_success
        and seed_pack.screening_rules.seed_roles["ranking_weakness"].lane == contract.role_seed("ranking_weakness").lane
        and seed_pack.screening_rules.seed_roles["ranking_weakness"].seed
        == contract.role_seed("ranking_weakness").seed
        and seed_pack.screening_rules.seed_roles["ranking_weakness"].required_min_success
        == contract.weakness_min_success_exclusive
        and seed_pack.screening_rules.seed_roles["guardrail"].lane == contract.role_seed("guardrail").lane
        and seed_pack.screening_rules.seed_roles["guardrail"].seed == contract.role_seed("guardrail").seed
        and seed_pack.screening_rules.seed_roles["guardrail"].required_min_success == contract.guardrail_min_success
        and seed_pack.screening_rules.seed_roles["sentinel"].lane == contract.role_seed("sentinel").lane
        and seed_pack.screening_rules.seed_roles["sentinel"].seed == contract.role_seed("sentinel").seed
        and seed_pack.screening_rules.seed_roles["sentinel"].required_behavior == "track_only"
    )

    checks = [
        {
            "label": "active_candidate_round6",
            "status": (
                "pass"
                if seed_pack.benchmark.active_candidate_name == contract.benchmark.active_candidate == "round6"
                else "fail"
            ),
            "detail": (
                "seed_pack.active_candidate="
                f"{seed_pack.benchmark.active_candidate_name}, contract.active_candidate={contract.benchmark.active_candidate}"
            ),
        },
        {
            "label": "active_pack_matches_contract",
            "status": (
                "pass"
                if seed_pack.benchmark.active_candidate_pack
                == contract.benchmark.active_candidate_pack
                == EXPECTED_ACTIVE_PACK
                else "fail"
            ),
            "detail": (
                "seed_pack.active_candidate_pack="
                f"{seed_pack.benchmark.active_candidate_pack}, contract.active_candidate_pack={contract.benchmark.active_candidate_pack}"
            ),
        },
        {
            "label": "archived_pack_matches_contract",
            "status": (
                "pass"
                if seed_pack.benchmark.archived_frozen_pack
                == contract.benchmark.archived_frozen_pack
                == EXPECTED_ARCHIVED_PACK
                else "fail"
            ),
            "detail": (
                "seed_pack.archived_frozen_pack="
                f"{seed_pack.benchmark.archived_frozen_pack}, contract.archived_frozen_pack={contract.benchmark.archived_frozen_pack}"
            ),
        },
        {
            "label": "restart_default_round7",
            "status": (
                "pass"
                if seed_pack.frontier.restart_default == (contract.frontier_roles.default_restart_prior,) == ("round7",)
                else "fail"
            ),
            "detail": f"restart_default={seed_pack.frontier.restart_default}",
        },
        {
            "label": "reserve_priors_match_contract_roles",
            "status": "pass" if set(seed_pack.frontier.reserve_priors) == expected_reserve else "fail",
            "detail": f"reserve_priors={seed_pack.frontier.reserve_priors}",
        },
        {
            "label": "retired_priors_match_contract_roles",
            "status": (
                "pass" if set(seed_pack.frontier.retired_priors) == set(contract.frontier_roles.retired_priors) else "fail"
            ),
            "detail": f"retired_priors={seed_pack.frontier.retired_priors}",
        },
        {
            "label": "support_status_matches_contract",
            "status": "pass" if seed_pack.frontier.support_status == contract.support_status else "fail",
            "detail": f"support_status={seed_pack.frontier.support_status}",
        },
        {
            "label": "generated_from_active_pack_current",
            "status": (
                "pass"
                if seed_pack.generated_from.active_candidate_pack == contract.benchmark.active_candidate_pack
                else "fail"
            ),
            "detail": f"generated_from.active_candidate_pack={seed_pack.generated_from.active_candidate_pack}",
        },
        {
            "label": "seed_roles_match_contract_thresholds",
            "status": "pass" if seed_roles_match else "fail",
            "detail": (
                "support="
                f"{seed_pack.screening_rules.seed_roles['ranking_support']}, weakness={seed_pack.screening_rules.seed_roles['ranking_weakness']}, "
                f"guardrail={seed_pack.screening_rules.seed_roles['guardrail']}, sentinel={seed_pack.screening_rules.seed_roles['sentinel']}"
            ),
        },
        {
            "label": "validation_buckets_match_frontier_roles",
            "status": (
                "pass"
                if validation_restart_default == set(seed_pack.frontier.restart_default)
                and validation_reserve == set(seed_pack.frontier.reserve_priors)
                and validation_retired == set(seed_pack.frontier.retired_priors)
                and validation_bucket_by_candidate.get("round7") == "validated_restart_default"
                and validation_bucket_by_candidate.get("round10") == "validated_reserve"
                and validation_bucket_by_candidate.get("round5") == "validated_reserve"
                and validation_bucket_by_candidate.get("door3_post5") == "validated_retired"
                and validation_bucket_by_candidate.get("post_unlock_x5") == "validated_retired"
                and not validation.needs_review
                else "fail"
            ),
            "detail": (
                f"validated_restart_default={sorted(validation_restart_default)}, "
                f"validated_reserve={sorted(validation_reserve)}, "
                f"validated_retired={sorted(validation_retired)}, "
                f"needs_review={list(validation.needs_review)}"
            ),
        },
        {
            "label": "scorer_verdicts_match_frontier_roles",
            "status": (
                "pass"
                if scorer_verdict_by_candidate.get("round7") == "advance_for_broader_dev"
                and scorer_verdict_by_candidate.get("round10") == "advance_for_broader_dev"
                and scorer_verdict_by_candidate.get("round5") == "hold_seed_clean_but_below_incumbent"
                and scorer_verdict_by_candidate.get("door3_post5") == "prune_support_regression"
                and scorer_verdict_by_candidate.get("post_unlock_x5") == "prune_guardrail_regression"
                and scorer.incumbent_269 == contract.weakness_min_success_exclusive - 1e-9
                else "fail"
            ),
            "detail": f"scorer_grouped={scorer.grouped}",
        },
    ]
    overall = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {"overall": overall, "checks": checks}


def render_seed_pack_doctor(output: Path | None, json_output: Path | None) -> dict[str, Any]:
    validation = load_seed_pack_validation_report(SEED_PACK_VALIDATION_PATH)
    scorer = load_seed_pack_scorer_report(SEED_PACK_SCORER_PATH)
    result = evaluate_seed_pack_state(validation=validation, scorer=scorer)

    if output is not None:
        lines = [
            "# Portfolio Seed-Pack Doctor",
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
    parser = argparse.ArgumentParser(description="Doctor check for the portfolio seed-pack state")
    parser.add_argument("--output", required=False)
    parser.add_argument("--json", required=False)
    parser.add_argument("--fail-on-drift", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = render_seed_pack_doctor(
        Path(args.output) if args.output else None,
        Path(args.json) if args.json else None,
    )
    if args.fail_on_drift and result["overall"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
