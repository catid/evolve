from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from psmn_rl.analysis.benchmark_pack import sha256_path
from psmn_rl.analysis.lss_post_pass_campaign import _read_json, _write_json
from psmn_rl.analysis.portfolio_frontier_contract_loader import load_frontier_contract
from psmn_rl.utils.io import get_git_commit, get_git_dirty


ARCHIVED_FROZEN_PACK = Path("outputs/reports/frozen_benchmark_pack.json")
CURRENT_GATE_REFERENCE_PACK = Path("outputs/reports/round6_current_benchmark_pack.json")
LIVE_PORTFOLIO_PACK = Path("outputs/reports/portfolio_candidate_pack.json")
NEXT_MEGA_PACK = Path("outputs/reports/next_mega_portfolio_candidate_pack.json")
NEXT_MEGA_GATE_REPORT = Path("outputs/reports/next_mega_portfolio_gate_report.json")
NEXT_MEGA_DECISION_MEMO = Path("outputs/reports/next_mega_portfolio_decision_memo.md")
CLAIM_LEDGER = Path("outputs/reports/claim_ledger.md")
CLAIM_LEDGER_SNAPSHOT = Path("outputs/reports/claim_ledger_round6_current.md")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def snapshot_claim_ledger(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CLAIM_LEDGER, output)


def refresh_live_pack(
    *,
    source_pack: Path,
    gate_reference_pack: Path,
    output: Path,
) -> dict[str, Any]:
    source = _read_json(source_pack)
    repaired = deepcopy(source)
    gate_pack = _read_json(gate_reference_pack)
    repaired["frozen_pack_reference"] = {
        "path": str(gate_reference_pack),
        "sha256": sha256_path(gate_reference_pack),
        "claim_id": str(gate_pack["claim"]["id"]),
    }
    repaired["active_benchmark_state"]["active_pack_role"] = "narrowed_active_round6"
    repaired["active_benchmark_state"]["current_active_pack"] = {
        "path": str(source_pack),
        "sha256": sha256_path(source_pack),
    }
    repaired["active_benchmark_state"]["winner"] = "round6"
    repaired["active_benchmark_state"]["challenger_viable_pre_gate"] = False
    repaired["portfolio_campaign"]["winner"] = "round6"
    repaired["portfolio_campaign"]["challenger_viable_pre_gate"] = False
    repaired["portfolio_campaign"]["active_pack_before_program"] = {
        "path": str(source_pack),
        "sha256": sha256_path(source_pack),
    }
    repaired["portfolio_campaign"]["future_comparison_policy"] = {
        "active_canonical_pack": str(output),
        "archived_legacy_pack": str(ARCHIVED_FROZEN_PACK),
        "gate_reference_pack": str(gate_reference_pack),
    }
    repaired.setdefault("migration", {})
    repaired["migration"]["future_comparison_policy"] = {
        "active_canonical_pack": str(output),
        "archived_legacy_pack": str(ARCHIVED_FROZEN_PACK),
        "gate_reference_pack": str(gate_reference_pack),
    }
    repaired.setdefault("provenance", {})
    repaired["provenance"]["git_commit"] = get_git_commit()
    repaired["provenance"]["git_dirty"] = get_git_dirty()
    repaired["provenance"]["state_repair_notes"] = (
        "reconciled narrowed round6 active state after the next-mega portfolio campaign"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(repaired, indent=2, sort_keys=True), encoding="utf-8")
    return repaired


def render_state_reconciliation(output: Path) -> None:
    contract = load_frontier_contract()
    next_mega_gate = _read_json(NEXT_MEGA_GATE_REPORT) if NEXT_MEGA_GATE_REPORT.exists() else {}
    live_pack = _read_json(LIVE_PORTFOLIO_PACK) if LIVE_PORTFOLIO_PACK.exists() else {}
    lines = [
        "# Next-Round State Reconciliation",
        "",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- archived frozen pack: `{ARCHIVED_FROZEN_PACK}`",
        f"- repaired operational gate reference pack: `{CURRENT_GATE_REFERENCE_PACK}`",
        f"- live active benchmark pack: `{LIVE_PORTFOLIO_PACK}`",
        "",
        "## Reconciled Current Truth",
        "",
        "- `round6` remains the active winning DoorKey line.",
        "- The archived frozen pack remains the legacy provenance anchor and is no longer treated as the mutable live-state validator.",
        "- The broad 80-run next-mega portfolio result narrowed the internal benchmark/frontier state instead of strengthening it.",
        "- The public claim remains narrow: teacher-guided only, KL learner-state only, DoorKey only, external 64-episode evaluation only.",
        "",
        "## Why Repair Was Needed",
        "",
        f"- The latest completed campaign ended with gate verdict `{next_mega_gate.get('verdict', 'missing')}` because the archived frozen pack still sealed the live mutable `claim_ledger.md` path.",
        "- That made later accepted-state ledger updates look like provenance drift even when the benchmark result itself had not changed.",
        "",
        "## Repaired Operational Interpretation",
        "",
        f"- active benchmark: `{contract.benchmark.active_candidate}`",
        f"- default restart prior: `{contract.frontier_roles.default_restart_prior}`",
        f"- replay-validated alternate: `{contract.frontier_roles.replay_validated_alternate}`",
        f"- hold-only prior: `{list(contract.frontier_roles.hold_only_priors)}`",
        f"- retired priors: `{list(contract.frontier_roles.retired_priors)}`",
        f"- live pack candidate: `{live_pack.get('candidate_name', 'missing')}`",
        f"- live pack gate reference: `{live_pack.get('portfolio_campaign', {}).get('future_comparison_policy', {}).get('gate_reference_pack', 'missing')}`",
        "",
        "## Acceptance",
        "",
        "- The repo now has one coherent pre-challenger state: archived frozen baseline for provenance, repaired current gate reference for live validation, and a narrowed `round6` operational benchmark state.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_baseline_sync(output: Path) -> None:
    contract = load_frontier_contract()
    frozen_pack = _read_json(ARCHIVED_FROZEN_PACK)
    live_pack = _read_json(LIVE_PORTFOLIO_PACK)
    holdout = live_pack.get("portfolio_campaign", {}).get("holdout_summary", {})
    anti_regression = live_pack.get("portfolio_campaign", {}).get("anti_regression_summary", {})
    route_validation = live_pack.get("portfolio_campaign", {}).get("route_validation", {})
    stability = live_pack.get("portfolio_campaign", {}).get("stability_validation", {})
    lines = [
        "# Next-Round Baseline Sync",
        "",
        f"- archived frozen pack: `{ARCHIVED_FROZEN_PACK}`",
        f"- live active benchmark pack: `{LIVE_PORTFOLIO_PACK}`",
        f"- repaired current gate reference pack: `{CURRENT_GATE_REFERENCE_PACK}`",
        "",
        "## Archived Frozen Baseline",
        "",
        f"- retry-block KL learner-state `SARE` threshold: `{float(frozen_pack['thresholds']['retry_block_means']['kl_lss_sare']):.4f}`",
        f"- combined DoorKey KL learner-state `SARE` threshold: `{float(frozen_pack['thresholds']['combined_means']['kl_lss_sare']):.4f}`",
        "",
        "## Active Round6 Benchmark",
        "",
        f"- candidate: `{live_pack['candidate_name']}`",
        f"- retry-block KL learner-state `SARE` mean: `{float(live_pack['metrics']['retry_block']['kl_lss_sare']['mean']):.4f}`",
        f"- frozen-comparable combined KL learner-state `SARE` mean: `{float(live_pack['metrics']['combined']['kl_lss_sare']['mean']):.4f}`",
        f"- holdout SARE/token/single: `{float(holdout.get('round6_summary', {}).get('sare_mean', 0.0)):.4f}` / `{float(holdout.get('round6_summary', {}).get('token_mean', 0.0)):.4f}` / `{float(holdout.get('round6_summary', {}).get('single_mean', 0.0)):.4f}`",
        f"- healthy anti-regression SARE/token/single: `{float(anti_regression.get('round6_summary', {}).get('sare_mean', 0.0)):.4f}` / `{float(anti_regression.get('round6_summary', {}).get('token_mean', 0.0)):.4f}` / `{float(anti_regression.get('round6_summary', {}).get('single_mean', 0.0)):.4f}`",
        f"- route validation incumbent pass: `{route_validation.get('round6_pass')}`",
        f"- stability incumbent pass: `{stability.get('round6_pass')}`",
        "",
        "## Current Frontier Priors",
        "",
        f"- default restart prior: `{contract.frontier_roles.default_restart_prior}`",
        f"- replay-validated alternate: `{contract.frontier_roles.replay_validated_alternate}`",
        f"- hold-only priors: `{list(contract.frontier_roles.hold_only_priors)}`",
        f"- retired priors: `{list(contract.frontier_roles.retired_priors)}`",
        f"- seed roles: `sentinel={contract.role_seed('sentinel')}`, `support={contract.role_seed('ranking_support')}`, `weakness={contract.role_seed('ranking_weakness')}`, `guardrail={contract.role_seed('guardrail')}`",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_gate_repair(output: Path, *, validation_json: Path, gate_json: Path) -> None:
    validation = _read_json(validation_json)
    gate = _read_json(gate_json)
    next_mega_gate = _read_json(NEXT_MEGA_GATE_REPORT)
    lines = [
        "# Next-Round Gate Repair",
        "",
        f"- archived frozen pack remains untouched: `{ARCHIVED_FROZEN_PACK}`",
        f"- repaired current gate reference pack: `{CURRENT_GATE_REFERENCE_PACK}`",
        f"- live active benchmark pack: `{LIVE_PORTFOLIO_PACK}`",
        "",
        "## Cause",
        "",
        f"- old next-mega gate verdict: `{next_mega_gate.get('verdict', 'missing')}`",
        "- root cause: the archived frozen pack sealed the mutable live `claim_ledger.md` path, so later accepted-state ledger edits surfaced as frozen-pack provenance drift.",
        "",
        "## Repair",
        "",
        f"- current claim-ledger snapshot: `{CLAIM_LEDGER_SNAPSHOT}`",
        "- the repaired operational pack seals immutable current-state artifacts, including the snapshot ledger, instead of depending on the live mutable ledger path.",
        "",
        "## Validation",
        "",
        f"- repaired pack validation verdict: `{validation.get('verdict', 'missing')}`",
        f"- live active-pack gate verdict: `{gate.get('verdict', 'missing')}`",
        f"- live gate frozen-pack target: `{gate.get('frozen_pack', 'missing')}`",
        "",
        "## Interpretation",
        "",
        "- The archived frozen pack stays the provenance anchor for the original narrow DoorKey claim.",
        "- The repaired current gate reference pack is now the operational validator for the live `round6` benchmark state.",
        "- New challenger evidence can therefore be interpreted again without the old ledger-hash ambiguity.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair and render the narrowed current round6 benchmark state")
    sub = parser.add_subparsers(dest="command", required=True)

    snapshot = sub.add_parser("snapshot-claim-ledger")
    snapshot.add_argument("--output", required=True)

    refresh = sub.add_parser("refresh-live-pack")
    refresh.add_argument("--source-pack", required=True)
    refresh.add_argument("--gate-reference-pack", required=True)
    refresh.add_argument("--output", required=True)

    reconcile = sub.add_parser("state-reconciliation")
    reconcile.add_argument("--output", required=True)

    baseline = sub.add_parser("baseline-sync")
    baseline.add_argument("--output", required=True)

    repair = sub.add_parser("gate-repair")
    repair.add_argument("--validation-json", required=True)
    repair.add_argument("--gate-json", required=True)
    repair.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "snapshot-claim-ledger":
        snapshot_claim_ledger(Path(args.output))
        return
    if args.command == "refresh-live-pack":
        refresh_live_pack(
            source_pack=Path(args.source_pack),
            gate_reference_pack=Path(args.gate_reference_pack),
            output=Path(args.output),
        )
        return
    if args.command == "state-reconciliation":
        render_state_reconciliation(Path(args.output))
        return
    if args.command == "baseline-sync":
        render_baseline_sync(Path(args.output))
        return
    if args.command == "gate-repair":
        render_gate_repair(
            Path(args.output),
            validation_json=Path(args.validation_json),
            gate_json=Path(args.gate_json),
        )
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
