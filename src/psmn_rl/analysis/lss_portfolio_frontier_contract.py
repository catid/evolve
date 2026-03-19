from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _load_yaml, _read_json, _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty

MANIFEST = Path("outputs/reports/portfolio_frontier_manifest.json")
CONSISTENCY = Path("outputs/reports/portfolio_frontier_consistency.json")
SEED_PACK = Path("outputs/reports/portfolio_seed_pack.json")
SCREENING_SPEC = Path("outputs/reports/portfolio_screening_spec.json")


def _singleton(values: list[str]) -> str | None:
    return values[0] if len(values) == 1 else None


def render_contract(campaign: dict[str, Any], output: Path, md_output: Path | None) -> None:
    manifest = _read_json(MANIFEST)
    consistency = _read_json(CONSISTENCY)
    seed_pack = _read_json(SEED_PACK)
    screening = _read_json(SCREENING_SPEC)

    contract = {
        "schema_version": 1,
        "contract_type": "portfolio_frontier_contract",
        "campaign_name": campaign["name"],
        "provenance": {
            "git_commit": get_git_commit(),
            "git_dirty": get_git_dirty(),
        },
        "consistency": {
            "overall": consistency["overall"],
            "checked_labels": [check["label"] for check in consistency["checks"]],
        },
        "benchmark": {
            "active_candidate": _singleton(manifest["active"]),
            "active_candidate_pack": seed_pack["benchmark"]["active_candidate_pack"],
            "archived_frozen_pack": seed_pack["benchmark"]["archived_frozen_pack"],
        },
        "frontier_roles": {
            "default_restart_prior": _singleton(manifest["default_restart"]),
            "replay_validated_alternate": _singleton(manifest["replay_validated_alternates"]),
            "hold_only_priors": list(manifest["hold_only"]),
            "retired_priors": list(manifest["retired"]),
        },
        "seed_roles": screening["seed_roles"],
        "seed_labels": {
            "global_hard": list(screening["global_hard"]),
            "differentiator": list(screening["differentiator"]),
        },
        "screening_thresholds": {
            "support_min_success": seed_pack["screening_rules"]["seed_roles"]["ranking_support"]["required_min_success"],
            "weakness_min_success_exclusive": seed_pack["screening_rules"]["seed_roles"]["ranking_weakness"]["required_min_success"],
            "guardrail_min_success": seed_pack["screening_rules"]["seed_roles"]["guardrail"]["required_min_success"],
        },
        "frontier_support_status": seed_pack["frontier"]["support_status"],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, contract)

    if md_output is not None:
        lines = [
            "# Portfolio Frontier Contract",
            "",
            f"- source campaign: `{campaign['name']}`",
            f"- git commit: `{get_git_commit()}`",
            f"- git dirty: `{get_git_dirty()}`",
            f"- consistency overall: `{contract['consistency']['overall']}`",
            "",
            "## Benchmark State",
            "",
            f"- active benchmark: `{contract['benchmark']['active_candidate']}`",
            f"- active pack: `{contract['benchmark']['active_candidate_pack']}`",
            f"- archived frozen pack: `{contract['benchmark']['archived_frozen_pack']}`",
            "",
            "## Frontier Roles",
            "",
            f"- default restart prior: `{contract['frontier_roles']['default_restart_prior']}`",
            f"- replay-validated alternate: `{contract['frontier_roles']['replay_validated_alternate']}`",
            f"- hold-only priors: `{contract['frontier_roles']['hold_only_priors']}`",
            f"- retired priors: `{contract['frontier_roles']['retired_priors']}`",
            "",
            "## Seed Contract",
            "",
            f"- seed roles: `{contract['seed_roles']}`",
            f"- global-hard labels: `{contract['seed_labels']['global_hard']}`",
            f"- differentiator labels: `{contract['seed_labels']['differentiator']}`",
            f"- thresholds: `{contract['screening_thresholds']}`",
            f"- support status: `{contract['frontier_support_status']}`",
            "",
            "## Interpretation",
            "",
            "- This contract is the single authoritative snapshot of the current measured portfolio frontier.",
            "- It freezes the active benchmark, restart prior, measured alternate, retired set, and seed-level screening contract in one place.",
            "- Future bounded search or doc updates should be treated as drifting until this contract is intentionally refreshed.",
        ]
        md_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a single contract artifact for the measured portfolio frontier")
    parser.add_argument("--campaign-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--md-output", required=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign = _load_yaml(Path(args.campaign_config))
    render_contract(campaign, Path(args.output), Path(args.md_output) if args.md_output else None)


if __name__ == "__main__":
    main()
