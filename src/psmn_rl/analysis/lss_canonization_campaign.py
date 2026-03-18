from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psmn_rl.analysis.benchmark_pack import sha256_path
from psmn_rl.analysis.claim_gate import evaluate_pack_claim_gate_from_paths, render_pack_gate_report
from psmn_rl.analysis.lss_frozen_claim import REPRO_MODES, _evaluate_modes
from psmn_rl.analysis.lss_post_pass_campaign import (
    DISPLAY_NAMES,
    LABELS,
    _artifact,
    _float,
    _int,
    _lane_seed_list,
    _load_yaml,
    _read_csv_rows,
    _read_json,
    _stats,
    _summary_round,
    _write_csv,
    _write_json,
)
from psmn_rl.analysis.lss_robustness import EvalTarget, _evaluate_targets, _format_float
from psmn_rl.utils.io import get_git_commit, get_git_dirty


CURRENT_CONTROLS: tuple[str, ...] = ("kl_lss_sare", "kl_lss_token_dense", "kl_lss_single_expert")
HARD_BLOCK_LABELS: tuple[str, ...] = ("post_pass_b", "post_pass_c")


@dataclass(slots=True)
class RunRecord:
    candidate: str
    lane: str
    seed: int
    label: str
    run_dir: Path
    summary: dict[str, Any]


def _discover_runs(root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    if not root.exists():
        return records
    for summary_path in sorted(root.glob("*/*/seed_*/kl_lss_*/summary.json")):
        run_dir = summary_path.parent
        seed_dir = run_dir.parent
        lane_dir = seed_dir.parent
        candidate_dir = lane_dir.parent
        if not seed_dir.name.startswith("seed_"):
            continue
        records.append(
            RunRecord(
                candidate=candidate_dir.name,
                lane=lane_dir.name,
                seed=int(seed_dir.name.split("_", 1)[1]),
                label=run_dir.name,
                run_dir=run_dir,
                summary=_read_json(summary_path),
            )
        )
    return records


def _candidate_row(record: RunRecord) -> dict[str, Any]:
    best_round_index = _int(record.summary.get("best_round_index"))
    best_round = _summary_round(record.summary, best_round_index)
    final_round = _summary_round(record.summary, len(record.summary.get("rounds", [])))
    return {
        "candidate": record.candidate,
        "lane": record.lane,
        "seed": record.seed,
        "label": record.label,
        "run_dir": str(record.run_dir),
        "config_path": str(record.run_dir / "student_resolved_config.yaml"),
        "checkpoint_path": str(record.run_dir / "latest.pt"),
        "final_greedy_success": _float(record.summary.get("final_greedy_success")),
        "best_round_index": best_round_index,
        "best_round_greedy_success": _float(record.summary.get("best_round_greedy_success")),
        "best_round_disagreement": _float(best_round.get("collection/disagreement_rate")),
        "best_round_unique_ratio": _float(best_round.get("collection/unique_state_ratio")),
        "best_round_post_unlock_frac": _float(best_round.get("collection/phase_frac_post_unlock")),
        "final_round_disagreement": _float(final_round.get("collection/disagreement_rate")),
        "final_round_unique_ratio": _float(final_round.get("collection/unique_state_ratio")),
        "final_round_post_unlock_frac": _float(final_round.get("collection/phase_frac_post_unlock")),
        "final_round_route_entropy": _float(final_round.get("collection/route_entropy")),
        "final_round_path_entropy": _float(final_round.get("collection/path_entropy")),
        "final_round_active_compute": _float(final_round.get("collection/active_compute_proxy")),
    }


def _rows_to_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    return {
        (str(row["candidate"]), str(row["lane"]), int(row["seed"]), str(row["label"])): row
        for row in rows
    }


def _label_rows(rows: list[dict[str, Any]], candidate: str, label: str, lanes: set[str] | None = None) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row["candidate"]) == candidate and str(row["label"]) == label]
    if lanes is not None:
        selected = [row for row in selected if str(row["lane"]) in lanes]
    return selected


def _block_summary_from_rows(rows: list[dict[str, Any]], candidate: str) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for lane in HARD_BLOCK_LABELS:
        block_rows = [row for row in rows if str(row["candidate"]) == candidate and str(row["lane"]) == lane]
        if not block_rows:
            continue
        label_stats = {
            label: _stats([_float(row["final_greedy_success"]) for row in block_rows if str(row["label"]) == label])
            for label in CURRENT_CONTROLS
        }
        summary.append(
            {
                "lane": lane,
                "candidate_mean": label_stats["kl_lss_sare"]["mean"],
                "token_mean": label_stats["kl_lss_token_dense"]["mean"],
                "single_mean": label_stats["kl_lss_single_expert"]["mean"],
                "candidate_minus_token": label_stats["kl_lss_sare"]["mean"] - label_stats["kl_lss_token_dense"]["mean"],
                "candidate_minus_single": label_stats["kl_lss_sare"]["mean"] - label_stats["kl_lss_single_expert"]["mean"],
                "candidate_failures": label_stats["kl_lss_sare"]["complete_seed_failures"],
            }
        )
    return summary


def _hard_family_stats(rows: list[dict[str, Any]], candidate: str, label: str) -> dict[str, float]:
    values = [
        _float(row["final_greedy_success"])
        for row in rows
        if str(row["candidate"]) == candidate and str(row["label"]) == label and str(row["lane"]) in HARD_BLOCK_LABELS
    ]
    return _stats(values)


def _strong_family_stats(rows: list[dict[str, Any]], label: str) -> dict[str, float]:
    return _stats([_float(row["final_greedy_success"]) for row in rows if str(row["label"]) == label])


def _current_candidate_run_dir(campaign: dict[str, Any], lane: str, seed: int) -> Path:
    current_name = str(campaign["current_candidate_name"])
    if lane == "post_pass_b":
        return Path(campaign["existing_lane_roots"]["post_pass_b"]["root"]) / f"seed_{seed}" / "kl_lss_sare"
    if lane == "post_pass_c":
        return Path(campaign["stage_roots"]["stage2_candidates"]) / current_name / lane / f"seed_{seed}" / "kl_lss_sare"
    if lane == "fresh_final":
        return Path(campaign["existing_lane_roots"]["retry_block"]["root"]) / f"seed_{seed}" / "kl_lss_sare"
    return Path(campaign["existing_lane_roots"]["strong_candidate"]["root"]) / lane / f"seed_{seed}" / "kl_lss_sare"


def _best_candidate_run_dir(campaign: dict[str, Any], candidate: str, lane: str, seed: int) -> Path:
    if lane in HARD_BLOCK_LABELS:
        return Path(campaign["stage_roots"]["stage3_fairness"]) / candidate / lane / f"seed_{seed}" / "kl_lss_sare"
    if lane == "fresh_final":
        return Path(campaign["stage_roots"]["stage6_retry_pack"]) / candidate / lane / f"seed_{seed}" / "kl_lss_sare"
    return Path(campaign["stage_roots"]["stage4_replication"]) / candidate / lane / f"seed_{seed}" / "kl_lss_sare"


def _summary_compare_row(campaign: dict[str, Any], candidate: str, lane: str, seed: int) -> dict[str, Any]:
    candidate_summary = _read_json(_best_candidate_run_dir(campaign, candidate, lane, seed) / "summary.json")
    baseline_summary = _read_json(_current_candidate_run_dir(campaign, lane, seed) / "summary.json")
    candidate_best = _summary_round(candidate_summary, _int(candidate_summary.get("best_round_index")))
    baseline_best = _summary_round(baseline_summary, _int(baseline_summary.get("best_round_index")))
    candidate_final = _summary_round(candidate_summary, len(candidate_summary.get("rounds", [])))
    baseline_final = _summary_round(baseline_summary, len(baseline_summary.get("rounds", [])))
    return {
        "candidate_best_round_index": _int(candidate_summary.get("best_round_index")),
        "baseline_best_round_index": _int(baseline_summary.get("best_round_index")),
        "candidate_best_round_greedy_success": _float(candidate_summary.get("best_round_greedy_success")),
        "baseline_best_round_greedy_success": _float(baseline_summary.get("best_round_greedy_success")),
        "candidate_final_greedy_success": _float(candidate_summary.get("final_greedy_success")),
        "baseline_final_greedy_success": _float(baseline_summary.get("final_greedy_success")),
        "candidate_final_disagreement": _float(candidate_final.get("collection/disagreement_rate")),
        "baseline_final_disagreement": _float(baseline_final.get("collection/disagreement_rate")),
        "candidate_final_post_unlock_frac": _float(candidate_final.get("collection/phase_frac_post_unlock")),
        "baseline_final_post_unlock_frac": _float(baseline_final.get("collection/phase_frac_post_unlock")),
        "candidate_final_route_entropy": _float(candidate_final.get("collection/route_entropy")),
        "baseline_final_route_entropy": _float(baseline_final.get("collection/route_entropy")),
        "candidate_final_path_entropy": _float(candidate_final.get("collection/path_entropy")),
        "baseline_final_path_entropy": _float(baseline_final.get("collection/path_entropy")),
        "candidate_final_active_compute": _float(candidate_final.get("collection/active_compute_proxy")),
        "baseline_final_active_compute": _float(baseline_final.get("collection/active_compute_proxy")),
        "candidate_best_round_post_unlock_frac": _float(candidate_best.get("collection/phase_frac_post_unlock")),
        "baseline_best_round_post_unlock_frac": _float(baseline_best.get("collection/phase_frac_post_unlock")),
    }


def _hard_block_display_seed_list(campaign: dict[str, Any]) -> list[str]:
    display: list[str] = []
    for block in campaign["seed_groups"]["hard_block_family"]["blocks"]:
        display.append(f"{block['lane']}={block['seeds']}")
    return display


def _render_registration(campaign: dict[str, Any], output: Path) -> None:
    lines = [
        "# Canonization Campaign Registration",
        "",
        f"- frozen benchmark pack: `{campaign['frozen_pack']}`",
        f"- current thaw-qualified candidate pack: `{campaign['current_candidate_pack']}`",
        f"- current candidate: `{campaign['current_candidate_name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Hard-Block Family",
        "",
        f"- existing hard block: `{campaign['seed_groups']['hard_block_existing']['lane']}` seeds `{campaign['seed_groups']['hard_block_existing']['seeds']}`",
        f"- new hard block: `{campaign['seed_groups']['hard_block_new']['lane']}` seeds `{campaign['seed_groups']['hard_block_new']['seeds']}`",
        "- `post_pass_c` uses the next unused three-prime fresh DoorKey block after `post_pass_b`, so the added block stays in the same seed-pattern class rather than broadening tasks.",
        "",
        "## Stage Gates",
        "",
        "- Stage 2: a new candidate must materially improve over the thaw-qualified incumbent on the hard-block family, narrow the token_dense gap, and add no new complete-seed failures.",
        "- Stage 3: on the hard-block family, `SARE` must no longer trail matched `token_dense` and must remain at least competitive with matched `single_expert`.",
        "- Stage 4: the best hard-block candidate must not materially regress the existing strong/recovered DoorKey blocks or introduce new complete-seed failures there.",
        "- Stage 5: the selected hard-block win must remain routing-dependent and avoid narrow checkpoint-spike behavior.",
        "- Stage 6: the successor pack must stay frozen-comparable, clear the existing pack-based gate, and be stronger than the current thaw-qualified pack rather than merely different.",
        "",
        "## Reports",
        "",
        f"- registration: `{campaign['reports']['registration']}`",
        f"- baseline sync: `{campaign['reports']['baseline_sync']}`",
        f"- shortlist: `{campaign['reports']['stage1_shortlist']}`",
        f"- screening: `{campaign['reports']['stage2_report']}`",
        f"- fairness: `{campaign['reports']['stage3_report']}`",
        f"- anti-regression: `{campaign['reports']['stage4_report']}`",
        f"- route/stability: `{campaign['reports']['stage5_report']}`",
        f"- successor pack: `{campaign['reports']['successor_pack_markdown']}`",
        f"- canonization decision: `{campaign['reports']['decision_memo']}`",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_baseline_sync(campaign: dict[str, Any], output: Path) -> None:
    manifest = _load_yaml(Path(campaign["frozen_manifest"]))
    frozen_pack = _read_json(Path(campaign["frozen_pack"]))
    current_pack = _read_json(Path(campaign["current_candidate_pack"]))
    stage1_rows = _read_csv_rows(Path(campaign["current_candidate_stage1_csv"]))
    block_rows = [row for row in stage1_rows if str(row["lane"]) in {"post_pass_a", "post_pass_b"} and str(row["label"]) in CURRENT_CONTROLS]
    current_retry = current_pack["metrics"]["retry_block"]["kl_lss_sare"]["mean"]
    current_single = current_pack["metrics"]["retry_block"]["kl_lss_single_expert"]["mean"]
    current_combined = current_pack["metrics"]["combined"]["kl_lss_sare"]["mean"]
    lines = [
        "# Canonization Baseline Sync",
        "",
        f"- frozen benchmark pack: `{campaign['frozen_pack']}`",
        f"- current thaw-qualified candidate pack: `{campaign['current_candidate_pack']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Frozen Thresholds",
        "",
        f"- retry-block KL learner-state `SARE` mean to beat: `{manifest['thresholds']['retry_block_means']['kl_lss_sare']:.4f}`",
        f"- retry-block KL learner-state `single_expert` mean to match or beat: `{manifest['thresholds']['retry_block_means']['kl_lss_single_expert']:.4f}`",
        f"- combined KL learner-state `SARE` mean to preserve: `{manifest['thresholds']['combined_means']['kl_lss_sare']:.4f}`",
        "",
        "## Current Thaw-Qualified Candidate Snapshot",
        "",
        f"- retry-block KL learner-state `SARE` mean: `{current_retry:.4f}`",
        f"- retry-block KL learner-state `single_expert` mean: `{current_single:.4f}`",
        f"- frozen-comparable combined KL learner-state `SARE` mean: `{current_combined:.4f}`",
        f"- frozen pack schema version: `{frozen_pack.get('schema_version')}`",
        "",
        "## Current Post-PASS Blocks",
        "",
        "| Block | Seed | Variant | Greedy Success |",
        "| --- | --- | --- | ---: |",
    ]
    for row in sorted(block_rows, key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"]))):
        lines.append(
            f"| {row['lane']} | {row['seed']} | {DISPLAY_NAMES[str(row['label'])]} | {_format_float(_float(row['final_greedy_success']))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The accepted starting point remains the thaw-qualified `post_unlock_weighted` candidate relative to the frozen benchmark pack.",
            "- The structural canonization blocker is unchanged at the start of this campaign: `post_pass_b` still leaves matched `token_dense` above the candidate even though the candidate clears the frozen-pack gate.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_mechanism_shortlist(campaign: dict[str, Any], output: Path) -> None:
    lines = [
        "# Canonization Stage 1 Mechanism Shortlist",
        "",
        "- blocker: `post_pass_b` stays token-dense-led even after the thaw-qualified `post_unlock_weighted` intervention",
        "- hard-block family for this campaign: `" + ", ".join(_hard_block_display_seed_list(campaign)) + "`",
        "",
        "| Candidate | Mechanism Hypothesis | Hard-Block Target | How It Could Fail | Strong-Block Risk |",
        "| --- | --- | --- | --- | --- |",
    ]
    for candidate_id, candidate in campaign["candidates"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{candidate_id}`",
                    str(candidate["hypothesis"]),
                    f"`{candidate['targets']}`",
                    str(candidate["failure_mode"]),
                    str(candidate["antiregression_risk"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- All shortlisted interventions stay inside the existing teacher-guided KL learner-state family.",
            "- The shortlist is deliberately bounded to post-unlock weighting, disagreement-aware weighting, and one cleanup-round extension rather than a broad knob search.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stage2_pass(summary: dict[str, Any], incumbent_failures: int) -> bool:
    return (
        summary["candidate_mean"] >= summary["incumbent_mean"] + 0.02
        and summary["gap_narrowing_vs_token"] >= 0.05
        and int(summary["candidate_failures"]) <= incumbent_failures
        and (summary["candidate_mean"] - summary["incumbent_mean"] >= 0.05 or summary["post_pass_b_delta"] >= 0.05)
    )


def _render_stage2(campaign: dict[str, Any], records: list[RunRecord], output: Path, csv_output: Path, json_output: Path) -> None:
    detail_rows = [_candidate_row(record) for record in records]
    lookup = _rows_to_lookup(detail_rows)
    incumbent = str(campaign["current_candidate_name"])
    control_rows = [row for row in detail_rows if str(row["candidate"]) == incumbent and str(row["label"]) in CURRENT_CONTROLS]
    control_lookup = {
        (str(row["lane"]), int(row["seed"]), str(row["label"])): row for row in control_rows
    }
    candidate_rows = [row for row in detail_rows if str(row["candidate"]) != incumbent and str(row["label"]) == "kl_lss_sare"]
    screening_rows: list[dict[str, Any]] = []
    for row in sorted(candidate_rows, key=lambda item: (str(item["candidate"]), str(item["lane"]), int(item["seed"]))):
        incumbent_row = control_lookup[(str(row["lane"]), int(row["seed"]), "kl_lss_sare")]
        token_row = control_lookup[(str(row["lane"]), int(row["seed"]), "kl_lss_token_dense")]
        single_row = control_lookup[(str(row["lane"]), int(row["seed"]), "kl_lss_single_expert")]
        screening_rows.append(
            {
                **row,
                "current_candidate_greedy_success": _float(incumbent_row["final_greedy_success"]),
                "current_token_dense_greedy_success": _float(token_row["final_greedy_success"]),
                "current_single_expert_greedy_success": _float(single_row["final_greedy_success"]),
                "delta_vs_current_candidate": _float(row["final_greedy_success"]) - _float(incumbent_row["final_greedy_success"]),
                "delta_vs_current_token_dense": _float(row["final_greedy_success"]) - _float(token_row["final_greedy_success"]),
                "delta_vs_current_single_expert": _float(row["final_greedy_success"]) - _float(single_row["final_greedy_success"]),
            }
        )
    incumbent_family = _hard_family_stats(detail_rows, incumbent, "kl_lss_sare")
    token_family = _hard_family_stats(detail_rows, incumbent, "kl_lss_token_dense")
    single_family = _hard_family_stats(detail_rows, incumbent, "kl_lss_single_expert")
    incumbent_failures = int(incumbent_family["complete_seed_failures"])

    candidate_summaries: list[dict[str, Any]] = []
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in screening_rows:
        by_candidate.setdefault(str(row["candidate"]), []).append(row)
    for candidate, rows in sorted(by_candidate.items()):
        candidate_stats = _stats([_float(row["final_greedy_success"]) for row in rows])
        block_means: dict[str, float] = {}
        for lane in HARD_BLOCK_LABELS:
            lane_rows = [row for row in rows if str(row["lane"]) == lane]
            block_means[lane] = _stats([_float(row["final_greedy_success"]) for row in lane_rows])["mean"]
        summary = {
            "candidate": candidate,
            "candidate_mean": candidate_stats["mean"],
            "incumbent_mean": incumbent_family["mean"],
            "token_mean": token_family["mean"],
            "single_mean": single_family["mean"],
            "candidate_failures": candidate_stats["complete_seed_failures"],
            "candidate_minus_token": candidate_stats["mean"] - token_family["mean"],
            "candidate_minus_single": candidate_stats["mean"] - single_family["mean"],
            "improvement_vs_current": candidate_stats["mean"] - incumbent_family["mean"],
            "gap_narrowing_vs_token": (candidate_stats["mean"] - token_family["mean"]) - (incumbent_family["mean"] - token_family["mean"]),
            "post_pass_b_delta": block_means.get("post_pass_b", 0.0) - _stats(
                [_float(row["final_greedy_success"]) for row in control_rows if str(row["lane"]) == "post_pass_b" and str(row["label"]) == "kl_lss_sare"]
            )["mean"],
        }
        summary["stage2_pass"] = _stage2_pass(summary, incumbent_failures)
        candidate_summaries.append(summary)
    advancing_candidates = [
        row["candidate"]
        for row in sorted(
            [row for row in candidate_summaries if row["stage2_pass"]],
            key=lambda item: (item["candidate_mean"], item["gap_narrowing_vs_token"]),
            reverse=True,
        )[:2]
    ]

    lines = [
        "# Canonization Stage 2 Hard-Block Screening",
        "",
        f"- incumbent hard-block family mean: `{incumbent_family['mean']:.4f}`",
        f"- incumbent matched hard-block token_dense mean: `{token_family['mean']:.4f}`",
        f"- incumbent matched hard-block single_expert mean: `{single_family['mean']:.4f}`",
        "",
        "| Candidate | Block | Seed | Final Greedy | Δ vs Current SARE | Δ vs Current token_dense | Δ vs Current single_expert | Best Round | Best-Round Disagreement |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in screening_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['candidate']}`",
                    str(row["lane"]),
                    str(row["seed"]),
                    _format_float(row["final_greedy_success"]),
                    _format_float(row["delta_vs_current_candidate"]),
                    _format_float(row["delta_vs_current_token_dense"]),
                    _format_float(row["delta_vs_current_single_expert"]),
                    str(_int(row["best_round_index"])),
                    _format_float(row["best_round_disagreement"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Candidate Summary",
            "",
            "| Candidate | Hard-Family Mean | Δ vs Current SARE | Gap Narrowing vs token_dense | Candidate - token_dense | Candidate - single_expert | Complete-Seed Failures | Packability | Stage 2 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in sorted(candidate_summaries, key=lambda item: item["candidate_mean"], reverse=True):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['candidate']}`",
                    _format_float(row["candidate_mean"]),
                    _format_float(row["improvement_vs_current"]),
                    _format_float(row["gap_narrowing_vs_token"]),
                    _format_float(row["candidate_minus_token"]),
                    _format_float(row["candidate_minus_single"]),
                    str(int(row["candidate_failures"])),
                    "ready" if row["stage2_pass"] else "hard_block_gap",
                    "`pass`" if row["stage2_pass"] else "`stop`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- advancing candidates: `{advancing_candidates}`",
            "- Stage 2 keeps only hard-block candidates that materially improve over the thaw-qualified incumbent, narrow the token_dense gap, and avoid new complete-seed failures.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(csv_output, screening_rows)
    _write_json(
        json_output,
        {
            "stage": "stage2",
            "incumbent_family": {
                "sare_mean": incumbent_family["mean"],
                "token_mean": token_family["mean"],
                "single_mean": single_family["mean"],
                "candidate_failures": incumbent_failures,
            },
            "detail_rows": screening_rows,
            "control_rows": control_rows,
            "candidate_summaries": candidate_summaries,
            "advancing_candidates": advancing_candidates,
        },
    )


def _stage3_pass(block_summaries: list[dict[str, Any]], family_summary: dict[str, Any]) -> bool:
    post_pass_b_row = next((row for row in block_summaries if str(row["lane"]) == "post_pass_b"), None)
    if post_pass_b_row is None:
        return False
    block_single_ok = all(_float(row["candidate_minus_single"]) >= -0.05 for row in block_summaries)
    return (
        _float(post_pass_b_row["candidate_minus_token"]) >= 0.0
        and _float(family_summary["candidate_minus_token"]) >= 0.0
        and _float(family_summary["candidate_minus_single"]) >= -0.02
        and block_single_ok
        and int(family_summary["candidate_failures"]) == 0
    )


def _select_hard_case(stage2_payload: dict[str, Any], candidate: str) -> dict[str, Any] | None:
    rows = [row for row in stage2_payload.get("detail_rows", []) if str(row["candidate"]) == candidate]
    if not rows:
        return None
    selected = max(
        rows,
        key=lambda item: (
            _float(item["delta_vs_current_candidate"]),
            _float(item["final_greedy_success"]),
            -_float(item["delta_vs_current_token_dense"]),
        ),
    )
    return {"lane": str(selected["lane"]), "seed": int(selected["seed"])}


def _render_stage3(stage2_payload: dict[str, Any], records: list[RunRecord], output: Path, csv_output: Path, json_output: Path) -> None:
    detail_rows = [_candidate_row(record) for record in records]
    candidate_summaries: list[dict[str, Any]] = []
    for candidate in stage2_payload.get("advancing_candidates", []):
        candidate_rows = [row for row in detail_rows if str(row["candidate"]) == candidate]
        block_summaries = _block_summary_from_rows(candidate_rows, candidate)
        family_sare = _hard_family_stats(candidate_rows, candidate, "kl_lss_sare")
        family_token = _hard_family_stats(candidate_rows, candidate, "kl_lss_token_dense")
        family_single = _hard_family_stats(candidate_rows, candidate, "kl_lss_single_expert")
        family_summary = {
            "candidate": candidate,
            "candidate_mean": family_sare["mean"],
            "token_mean": family_token["mean"],
            "single_mean": family_single["mean"],
            "candidate_minus_token": family_sare["mean"] - family_token["mean"],
            "candidate_minus_single": family_sare["mean"] - family_single["mean"],
            "candidate_failures": family_sare["complete_seed_failures"],
        }
        family_summary["stage3_pass"] = _stage3_pass(block_summaries, family_summary)
        family_summary["block_summaries"] = block_summaries
        candidate_summaries.append(family_summary)
    survivors = [row["candidate"] for row in candidate_summaries if row["stage3_pass"]]
    best_candidate = None
    selected_hard_case = None
    if survivors:
        best_summary = max(
            [row for row in candidate_summaries if row["stage3_pass"]],
            key=lambda item: (item["candidate_mean"], item["candidate_minus_token"]),
        )
        best_candidate = str(best_summary["candidate"])
        selected_hard_case = _select_hard_case(stage2_payload, best_candidate)

    lines = [
        "# Canonization Stage 3 Hard-Block Fairness",
        "",
        "| Candidate | Block | Seed | Variant | Final Greedy | Best Round | Best-Round Greedy |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in sorted(detail_rows, key=lambda item: (str(item["candidate"]), str(item["lane"]), int(item["seed"]), str(item["label"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['candidate']}`",
                    str(row["lane"]),
                    str(row["seed"]),
                    DISPLAY_NAMES[str(row["label"])],
                    _format_float(row["final_greedy_success"]),
                    str(_int(row["best_round_index"])),
                    _format_float(row["best_round_greedy_success"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Candidate Summary",
            "",
            "| Candidate | Hard-Family SARE | Hard-Family token_dense | Hard-Family single_expert | post_pass_b SARE-token | Hard-Family SARE-token | Hard-Family SARE-single | Stage 3 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(candidate_summaries, key=lambda item: item["candidate_mean"], reverse=True):
        post_pass_b = next((item for item in row["block_summaries"] if str(item["lane"]) == "post_pass_b"), None)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['candidate']}`",
                    _format_float(row["candidate_mean"]),
                    _format_float(row["token_mean"]),
                    _format_float(row["single_mean"]),
                    _format_float(0.0 if post_pass_b is None else _float(post_pass_b["candidate_minus_token"])),
                    _format_float(row["candidate_minus_token"]),
                    _format_float(row["candidate_minus_single"]),
                    "`pass`" if row["stage3_pass"] else "`stop`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- surviving candidates: `{survivors}`",
            f"- best surviving candidate: `{best_candidate}`",
            f"- selected improved hard-block case for Stage 5: `{selected_hard_case}`",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(csv_output, detail_rows)
    _write_json(
        json_output,
        {
            "stage": "stage3",
            "detail_rows": detail_rows,
            "candidate_summaries": candidate_summaries,
            "surviving_candidates": survivors,
            "best_candidate": best_candidate,
            "selected_hard_case": selected_hard_case,
        },
    )


def _strong_block_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    blocks = ["original", "fresh", "fresh_extra"]
    for lane in blocks:
        block_rows = [row for row in rows if str(row["lane"]) == lane]
        if not block_rows:
            continue
        label_stats = {
            label: _stats([_float(row["final_greedy_success"]) for row in block_rows if str(row["label"]) == label])
            for label in CURRENT_CONTROLS
        }
        summary.append(
            {
                "lane": lane,
                "candidate_mean": label_stats["kl_lss_sare"]["mean"],
                "token_mean": label_stats["kl_lss_token_dense"]["mean"],
                "single_mean": label_stats["kl_lss_single_expert"]["mean"],
                "candidate_minus_token": label_stats["kl_lss_sare"]["mean"] - label_stats["kl_lss_token_dense"]["mean"],
                "candidate_minus_single": label_stats["kl_lss_sare"]["mean"] - label_stats["kl_lss_single_expert"]["mean"],
                "candidate_failures": label_stats["kl_lss_sare"]["complete_seed_failures"],
            }
        )
    return summary


def _render_stage4(campaign: dict[str, Any], stage3_payload: dict[str, Any], records: list[RunRecord], output: Path, csv_output: Path, json_output: Path) -> None:
    best_candidate = stage3_payload.get("best_candidate")
    if not best_candidate:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("# Canonization Stage 4 Anti-Regression\n\n- skipped: `no candidate survived hard-block fairness`\n", encoding="utf-8")
        _write_csv(csv_output, [])
        _write_json(
            json_output,
            {
                "stage": "stage4",
                "best_candidate": None,
                "detail_rows": [],
                "block_summaries": [],
                "stage4_pass": False,
                "selected_strong_case": None,
                "skipped": True,
            },
        )
        return

    detail_rows = [_candidate_row(record) for record in records if record.candidate == str(best_candidate)]
    current_rows = [
        row
        for row in _read_csv_rows(Path(campaign["current_candidate_stage2_csv"]))
        if str(row["block"]) in {"original", "fresh", "fresh_extra"} and str(row["label"]) in CURRENT_CONTROLS
    ]
    current_strong_sare = _stats([_float(row["final_greedy_success"]) for row in current_rows if str(row["label"]) == "kl_lss_sare"])
    candidate_strong_sare = _strong_family_stats(detail_rows, "kl_lss_sare")
    current_lookup = {
        (str(row["block"]), int(row["seed"]), str(row["label"])): row for row in current_rows
    }
    healthy_failures: list[tuple[str, int]] = []
    for row in detail_rows:
        if str(row["label"]) != "kl_lss_sare":
            continue
        current_row = current_lookup[(str(row["lane"]), int(row["seed"]), "kl_lss_sare")]
        if _float(current_row["final_greedy_success"]) > 0.0 and _float(row["final_greedy_success"]) <= 0.0:
            healthy_failures.append((str(row["lane"]), int(row["seed"])))
    stage4_pass = (
        candidate_strong_sare["mean"] >= current_strong_sare["mean"] - 0.02
        and not healthy_failures
        and candidate_strong_sare["complete_seed_failures"] <= current_strong_sare["complete_seed_failures"]
    )
    block_summaries = _strong_block_summary(detail_rows)
    strong_sare_rows = [row for row in detail_rows if str(row["label"]) == "kl_lss_sare"]
    selected_strong = None
    if strong_sare_rows:
        selected = max(strong_sare_rows, key=lambda item: (_float(item["final_greedy_success"]), -int(item["seed"])))
        selected_strong = {"lane": str(selected["lane"]), "seed": int(selected["seed"])}

    lines = [
        "# Canonization Stage 4 Anti-Regression",
        "",
        f"- best candidate: `{best_candidate}`",
        f"- current strong-block KL learner-state `SARE` mean: `{current_strong_sare['mean']:.4f}`",
        f"- candidate strong-block KL learner-state `SARE` mean: `{candidate_strong_sare['mean']:.4f}`",
        "",
        "| Lane | Seed | Variant | Final Greedy | Best Round | Best-Round Greedy |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in sorted(detail_rows, key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["lane"]),
                    str(row["seed"]),
                    DISPLAY_NAMES[str(row["label"])],
                    _format_float(row["final_greedy_success"]),
                    str(_int(row["best_round_index"])),
                    _format_float(row["best_round_greedy_success"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-Block Means",
            "",
            "| Block | Candidate SARE | Candidate token_dense | Candidate single_expert | Candidate - token_dense | Candidate - single_expert |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in block_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["lane"]),
                    _format_float(row["candidate_mean"]),
                    _format_float(row["token_mean"]),
                    _format_float(row["single_mean"]),
                    _format_float(row["candidate_minus_token"]),
                    _format_float(row["candidate_minus_single"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- new complete-seed failures on previously healthy strong seeds: `{healthy_failures}`",
            f"- selected strong-block case for Stage 5: `{selected_strong}`",
            f"- stage-4 status: `{'pass' if stage4_pass else 'stop'}`",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(csv_output, detail_rows)
    _write_json(
        json_output,
        {
            "stage": "stage4",
            "best_candidate": best_candidate,
            "detail_rows": detail_rows,
            "block_summaries": block_summaries,
            "current_strong_sare_mean": current_strong_sare["mean"],
            "candidate_strong_sare_mean": candidate_strong_sare["mean"],
            "healthy_failures": healthy_failures,
            "selected_strong_case": selected_strong,
            "stage4_pass": stage4_pass,
        },
    )


def _stability_targets(campaign: dict[str, Any], candidate: str, selected_hard_case: dict[str, Any], selected_strong_case: dict[str, Any]) -> tuple[list[EvalTarget], list[dict[str, Any]]]:
    targets: list[EvalTarget] = []
    cases = [
        {"kind": "hard", "lane": str(selected_hard_case["lane"]), "seed": int(selected_hard_case["seed"])},
        {"kind": "strong", "lane": str(selected_strong_case["lane"]), "seed": int(selected_strong_case["seed"])},
    ]
    for case in cases:
        lane = str(case["lane"])
        seed = int(case["seed"])
        for source in ("candidate", "baseline"):
            run_dir = _best_candidate_run_dir(campaign, candidate, lane, seed) if source == "candidate" else _current_candidate_run_dir(campaign, lane, seed)
            summary = _read_json(run_dir / "summary.json")
            config_path = run_dir / "student_resolved_config.yaml"
            rounds = len(summary.get("rounds", []))
            for round_index in range(1, rounds + 1):
                targets.append(
                    EvalTarget(
                        seed=seed,
                        label=f"{source}_{lane}_{seed}_round_{round_index}",
                        variant="sare",
                        config_path=config_path,
                        checkpoint_path=run_dir / f"round_{round_index:02d}.pt",
                        run_dir=run_dir,
                        method=f"{source}_round_eval",
                        stage="canonization_stage5",
                        round_index=round_index,
                        metadata={"lane": lane, "source": source, "case_kind": str(case["kind"])},
                        command_path=run_dir / "command.txt",
                    )
                )
    return targets, cases


def _stability_class(values: list[float]) -> str:
    if not values:
        return "no_data"
    sorted_values = sorted(values, reverse=True)
    best = sorted_values[0]
    second = sorted_values[1] if len(sorted_values) > 1 else sorted_values[0]
    final_value = values[-1]
    if (best - second) <= 0.125 and (best - final_value) <= 0.125:
        return "stable_plateau"
    if (best - second) > 0.25 or (best - final_value) > 0.25:
        return "narrow_spike"
    return "noisy_brittle"


def _render_stage5(campaign: dict[str, Any], stage3_payload: dict[str, Any], stage4_payload: dict[str, Any], route_rows: list[dict[str, Any]], device: str, episodes: int, output: Path, csv_output: Path, json_output: Path) -> None:
    best_candidate = stage3_payload.get("best_candidate")
    selected_hard_case = stage3_payload.get("selected_hard_case")
    selected_strong_case = stage4_payload.get("selected_strong_case")
    if not best_candidate or not selected_hard_case or not selected_strong_case:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("# Canonization Stage 5 Stability And Route\n\n- skipped: `candidate did not survive into stability and route validation`\n", encoding="utf-8")
        _write_csv(csv_output, [])
        _write_json(
            json_output,
            {
                "stage": "stage5",
                "best_candidate": best_candidate,
                "route_summaries": [],
                "stability_summaries": [],
                "stage5_pass": False,
                "skipped": True,
            },
        )
        return

    stability_targets, cases = _stability_targets(campaign, str(best_candidate), selected_hard_case, selected_strong_case)
    eval_rows = [row for row in _evaluate_targets(stability_targets, device, episodes) if str(row["mode"]) == "greedy"]
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in eval_rows:
        grouped.setdefault((str(row["source"]), str(row["lane"]), int(row["seed"])), []).append(row)
    stability_summaries: list[dict[str, Any]] = []
    for case in cases:
        lane = str(case["lane"])
        seed = int(case["seed"])
        for source in ("candidate", "baseline"):
            ordered = sorted(grouped.get((source, lane, seed), []), key=lambda item: int(item["round_index"]))
            successes = [_float(row["eval_success_rate"]) for row in ordered]
            stability_summaries.append(
                {
                    "source": source,
                    "lane": lane,
                    "seed": seed,
                    "case_kind": str(case["kind"]),
                    "round_successes": successes,
                    "classification": _stability_class(successes),
                }
            )

    route_grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in route_rows:
        route_grouped.setdefault((str(row["lane"]), int(row["seed"])), []).append(row)
    route_summaries: list[dict[str, Any]] = []
    for case in cases:
        lane = str(case["lane"])
        seed = int(case["seed"])
        case_rows = route_grouped.get((lane, seed), [])
        if not case_rows:
            continue
        baseline = next(row for row in case_rows if row["probe"] == "baseline")
        fixed = next(row for row in case_rows if row["probe"] == "router_override")
        random = next(row for row in case_rows if row["probe"] == "route_randomization")
        ablations = [row for row in case_rows if row["probe"] == "expert_ablation"]
        worst = min(ablations, key=lambda item: _float(item["eval_success_rate"])) if ablations else baseline
        compare = _summary_compare_row(campaign, str(best_candidate), lane, seed)
        route_summaries.append(
            {
                "lane": lane,
                "seed": seed,
                "case_kind": str(case["kind"]),
                "baseline_success": _float(baseline["eval_success_rate"]),
                "fixed_drop": _float(baseline["eval_success_rate"]) - _float(fixed["eval_success_rate"]),
                "random_drop": _float(baseline["eval_success_rate"]) - _float(random["eval_success_rate"]),
                "worst_ablation_drop": _float(baseline["eval_success_rate"]) - _float(worst["eval_success_rate"]),
                "route_entropy": _float(baseline.get("route_entropy")),
                "path_entropy": _float(baseline.get("path_entropy")),
                "active_compute_proxy": _float(baseline.get("active_compute_proxy")),
                **compare,
            }
        )
    candidate_stability = [row for row in stability_summaries if row["source"] == "candidate"]
    stability_ok = all(row["classification"] != "narrow_spike" for row in candidate_stability)
    fixed_ok = bool(route_summaries) and all(row["fixed_drop"] >= 0.25 for row in route_summaries)
    ablation_ok = bool(route_summaries) and all(row["worst_ablation_drop"] >= 0.25 for row in route_summaries)
    stage5_pass = stability_ok and fixed_ok and ablation_ok

    lines = [
        "# Canonization Stage 5 Stability And Route",
        "",
        f"- best candidate: `{best_candidate}`",
        "",
        "## Stability",
        "",
        "| Case | Lane | Seed | Source | Round Successes | Classification |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in stability_summaries:
        success_str = ", ".join(f"{value:.4f}" for value in row["round_successes"])
        lines.append(
            f"| {row['case_kind']} | {row['lane']} | {row['seed']} | {row['source']} | `{success_str}` | `{row['classification']}` |"
        )
    lines.extend(
        [
            "",
            "## Route Validation",
            "",
            "| Case | Lane | Seed | Baseline | Fixed-Router Drop | Route-Randomization Drop | Worst Ablation Drop | Final Disagreement Δ | Final Post-Unlock Frac Δ | Final Route Entropy Δ | Final Path Entropy Δ |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in route_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case_kind"]),
                    str(row["lane"]),
                    str(row["seed"]),
                    _format_float(row["baseline_success"]),
                    _format_float(row["fixed_drop"]),
                    _format_float(row["random_drop"]),
                    _format_float(row["worst_ablation_drop"]),
                    _format_float(row["candidate_final_disagreement"] - row["baseline_final_disagreement"]),
                    _format_float(row["candidate_final_post_unlock_frac"] - row["baseline_final_post_unlock_frac"]),
                    _format_float(row["candidate_final_route_entropy"] - row["baseline_final_route_entropy"]),
                    _format_float(row["candidate_final_path_entropy"] - row["baseline_final_path_entropy"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- stage-5 status: `{'pass' if stage5_pass else 'stop'}`",
            "- The route summary compares the best canonization candidate against the current thaw-qualified incumbent on the same seeds so a hard-block gain cannot hide by erasing route structure.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(csv_output, route_summaries)
    _write_json(
        json_output,
        {
            "stage": "stage5",
            "best_candidate": best_candidate,
            "selected_hard_case": selected_hard_case,
            "selected_strong_case": selected_strong_case,
            "route_summaries": route_summaries,
            "stability_summaries": stability_summaries,
            "stage5_pass": stage5_pass,
        },
    )


def _dedupe_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["lane"]), int(row["seed"]), str(row["label"]))
        deduped[key] = row
    return list(deduped.values())


def _pack_metric_rows(rows: list[dict[str, Any]], lane_seeds: list[tuple[str, int]]) -> list[dict[str, Any]]:
    lane_seed_set = {(lane, seed) for lane, seed in lane_seeds}
    return [row for row in rows if (str(row["lane"]), int(row["seed"])) in lane_seed_set]


def _metric_block(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {label: _stats([_float(row["eval_success_rate"]) for row in rows if str(row["label"]) == label]) for label in LABELS}


def _candidate_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "lane": str(row["lane"]),
        "seed": int(row["seed"]),
        "label": str(row["label"]),
        "eval_success_rate": _float(row["final_greedy_success"]),
        "run_dir": str(row["run_dir"]),
    }


def _render_successor_pack(
    campaign: dict[str, Any],
    stage3_payload: dict[str, Any],
    stage4_payload: dict[str, Any],
    stage5_payload: dict[str, Any],
    retry_records: list[RunRecord],
    summary_md: Path,
    metrics_json: Path,
    combined_md: Path,
    combined_csv: Path,
    retry_md: Path,
    retry_csv: Path,
    successor_md: Path,
    successor_json: Path,
) -> None:
    best_candidate = str(stage3_payload["best_candidate"])
    baseline_combined_rows = [row for row in _read_csv_rows(Path(campaign["frozen_combined_csv"])) if str(row.get("mode")) == "greedy"]
    baseline_rows = [
        {
            "lane": str(row["lane"]),
            "seed": int(row["seed"]),
            "label": str(row["label"]),
            "eval_success_rate": _float(row["eval_success_rate"]),
            "run_dir": str(row["run_dir"]),
        }
        for row in baseline_combined_rows
        if str(row["label"]) in {"recovered_token_dense", "baseline_sare"}
    ]
    stage4_rows = [_candidate_metric_row(row) for row in stage4_payload.get("detail_rows", []) if str(row["label"]) in CURRENT_CONTROLS]
    retry_rows = [_candidate_metric_row(_candidate_row(record)) for record in retry_records if record.candidate == best_candidate]
    candidate_rows = _dedupe_metric_rows(stage4_rows + retry_rows)
    combined_lane_seeds = _lane_seed_list(campaign["seed_groups"]["frozen_combined"])
    retry_lane_seeds = _lane_seed_list(campaign["seed_groups"]["retry_block"])
    combined_rows = _dedupe_metric_rows(_pack_metric_rows(candidate_rows + baseline_rows, combined_lane_seeds))
    retry_rows_pack = _dedupe_metric_rows(_pack_metric_rows(candidate_rows + baseline_rows, retry_lane_seeds))
    combined_metrics = _metric_block(combined_rows)
    retry_metrics = _metric_block(retry_rows_pack)

    combined_lines = [
        "# Canonization Candidate Frozen-Comparable Combined Report",
        "",
        f"- candidate: `{best_candidate}`",
        "",
        "| Lane | Seed | Variant | Greedy Success |",
        "| --- | --- | --- | ---: |",
    ]
    for row in sorted(combined_rows, key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"]))):
        combined_lines.append(
            f"| {row['lane']} | {row['seed']} | {DISPLAY_NAMES[str(row['label'])]} | {_format_float(_float(row['eval_success_rate']))} |"
        )
    combined_lines.extend(["", "## Summary", "", "| Variant | Mean | Complete-Seed Failures |", "| --- | ---: | ---: |"])
    for label in LABELS:
        combined_lines.append(
            f"| {DISPLAY_NAMES[label]} | `{combined_metrics[label]['mean']:.4f}` | `{int(combined_metrics[label]['complete_seed_failures'])}` |"
        )
    combined_md.parent.mkdir(parents=True, exist_ok=True)
    combined_md.write_text("\n".join(combined_lines) + "\n", encoding="utf-8")
    _write_csv(combined_csv, combined_rows)

    retry_lines = [
        "# Canonization Candidate Retry-Block Report",
        "",
        "| Seed | recovered token_dense | KL learner-state token_dense | KL learner-state single_expert | baseline PPO SARE | KL learner-state SARE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lookup = {(str(row["lane"]), int(row["seed"]), str(row["label"])): row for row in retry_rows_pack}
    for _lane, seed in retry_lane_seeds:
        retry_lines.append(
            "| "
            + " | ".join(
                [
                    str(seed),
                    _format_float(lookup[("fresh_final", seed, "recovered_token_dense")]["eval_success_rate"]),
                    _format_float(lookup[("fresh_final", seed, "kl_lss_token_dense")]["eval_success_rate"]),
                    _format_float(lookup[("fresh_final", seed, "kl_lss_single_expert")]["eval_success_rate"]),
                    _format_float(lookup[("fresh_final", seed, "baseline_sare")]["eval_success_rate"]),
                    _format_float(lookup[("fresh_final", seed, "kl_lss_sare")]["eval_success_rate"]),
                ]
            )
            + " |"
        )
    retry_lines.extend(["", "## Summary", "", "| Variant | Mean | Complete-Seed Failures |", "| --- | ---: | ---: |"])
    for label in LABELS:
        retry_lines.append(
            f"| {DISPLAY_NAMES[label]} | `{retry_metrics[label]['mean']:.4f}` | `{int(retry_metrics[label]['complete_seed_failures'])}` |"
        )
    retry_md.parent.mkdir(parents=True, exist_ok=True)
    retry_md.write_text("\n".join(retry_lines) + "\n", encoding="utf-8")
    _write_csv(retry_csv, retry_rows_pack)

    current_pack = _read_json(Path(campaign["current_candidate_pack"]))
    current_retry = _float(current_pack["metrics"]["retry_block"]["kl_lss_sare"]["mean"])
    current_combined = _float(current_pack["metrics"]["combined"]["kl_lss_sare"]["mean"])
    stage3_summary = next(row for row in stage3_payload["candidate_summaries"] if str(row["candidate"]) == best_candidate)
    summary_lines = [
        "# Canonization Candidate Summary",
        "",
        f"- candidate: `{best_candidate}`",
        f"- retry-block KL learner-state `SARE` mean: `{retry_metrics['kl_lss_sare']['mean']:.4f}`",
        f"- retry-block KL learner-state `single_expert` mean: `{retry_metrics['kl_lss_single_expert']['mean']:.4f}`",
        f"- frozen-comparable combined KL learner-state `SARE` mean: `{combined_metrics['kl_lss_sare']['mean']:.4f}`",
        f"- hard-block family KL learner-state `SARE` mean: `{stage3_summary['candidate_mean']:.4f}`",
        f"- hard-block family matched token_dense mean: `{stage3_summary['token_mean']:.4f}`",
        f"- retry-block SARE delta vs current thaw-qualified pack: `{retry_metrics['kl_lss_sare']['mean'] - current_retry:.4f}`",
        f"- combined SARE delta vs current thaw-qualified pack: `{combined_metrics['kl_lss_sare']['mean'] - current_combined:.4f}`",
    ]
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    metrics_payload = {
        "schema_version": 1,
        "candidate_name": best_candidate,
        "task": "DoorKey",
        "evaluation": campaign["evaluation"],
        "requested_claims": ["bounded_teacher_guided_doorkey_sare"],
        "controls_present": list(LABELS),
        "metrics": {"combined": combined_metrics, "retry_block": retry_metrics},
        "actual_sets": {
            "combined_lane_seeds": [[lane, seed] for lane, seed in combined_lane_seeds],
            "retry_block_lane_seeds": [[lane, seed] for lane, seed in retry_lane_seeds],
        },
        "provenance": {
            "git_commit": get_git_commit(),
            "git_dirty": get_git_dirty(),
            "notes": f"canonization successor candidate `{best_candidate}`",
        },
        "comparison_vs_current": {
            "current_candidate_name": str(current_pack.get("candidate_name", campaign["current_candidate_name"])),
            "retry_block_sare_delta": retry_metrics["kl_lss_sare"]["mean"] - current_retry,
            "combined_sare_delta": combined_metrics["kl_lss_sare"]["mean"] - current_combined,
        },
    }
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    frozen_pack_path = Path(campaign["frozen_pack"])
    frozen_pack = _read_json(frozen_pack_path)
    candidate_pack = {
        "schema_version": 1,
        "pack_type": "candidate_result_pack",
        "candidate_name": best_candidate,
        "frozen_pack_reference": {
            "path": str(frozen_pack_path),
            "sha256": sha256_path(frozen_pack_path),
            "claim_id": frozen_pack["claim"]["id"],
        },
        "task": "DoorKey",
        "evaluation": campaign["evaluation"],
        "requested_claims": ["bounded_teacher_guided_doorkey_sare"],
        "controls_present": list(LABELS),
        "metrics": {"combined": combined_metrics, "retry_block": retry_metrics},
        "actual_sets": {
            "combined_lane_seeds": [[lane, seed] for lane, seed in combined_lane_seeds],
            "retry_block_lane_seeds": [[lane, seed] for lane, seed in retry_lane_seeds],
        },
        "artifacts": [
            _artifact(summary_md, "candidate_summary_markdown"),
            _artifact(metrics_json, "candidate_metrics_json"),
            _artifact(combined_md, "combined_report_markdown"),
            _artifact(combined_csv, "combined_report_csv"),
            _artifact(retry_md, "retry_block_report_markdown"),
            _artifact(retry_csv, "retry_block_report_csv"),
        ],
        "provenance": metrics_payload["provenance"],
        "qualification": {
            "hard_block_family_sare_mean": stage3_summary["candidate_mean"],
            "hard_block_family_token_mean": stage3_summary["token_mean"],
            "route_validation_pass": bool(stage5_payload.get("stage5_pass")),
            "strong_block_sare_mean": _float(stage4_payload.get("candidate_strong_sare_mean")),
        },
    }
    successor_json.parent.mkdir(parents=True, exist_ok=True)
    successor_json.write_text(json.dumps(candidate_pack, indent=2, sort_keys=True), encoding="utf-8")

    successor_lines = [
        "# Canonization Successor Candidate Pack",
        "",
        f"- candidate: `{best_candidate}`",
        "- status: `successor candidate pack only; not yet canonical until the decision memo says so`",
        f"- current thaw-qualified pack: `{campaign['current_candidate_pack']}`",
        f"- frozen benchmark pack: `{campaign['frozen_pack']}`",
        "",
        "## Stronger-Than-Current Summary",
        "",
        f"- hard-block family KL learner-state `SARE` mean: `{stage3_summary['candidate_mean']:.4f}` vs matched token_dense `{stage3_summary['token_mean']:.4f}`",
        f"- retry-block KL learner-state `SARE` mean: `{retry_metrics['kl_lss_sare']['mean']:.4f}` vs current thaw-qualified pack `{current_retry:.4f}`",
        f"- frozen-comparable combined KL learner-state `SARE` mean: `{combined_metrics['kl_lss_sare']['mean']:.4f}` vs current thaw-qualified pack `{current_combined:.4f}`",
        f"- route/stability status: `{'pass' if stage5_payload.get('stage5_pass') else 'mixed'}`",
        "",
        "## Artifact Hashes",
        "",
        "| Role | Path | SHA256 |",
        "| --- | --- | --- |",
    ]
    for artifact in candidate_pack["artifacts"]:
        successor_lines.append(f"| {artifact['role']} | `{artifact['path']}` | `{artifact['sha256']}` |")
    successor_md.parent.mkdir(parents=True, exist_ok=True)
    successor_md.write_text("\n".join(successor_lines) + "\n", encoding="utf-8")


def _render_decision_memo(stage2_payload: dict[str, Any], stage3_payload: dict[str, Any] | None, stage4_payload: dict[str, Any] | None, stage5_payload: dict[str, Any] | None, gate_payload: dict[str, Any] | None, output: Path) -> None:
    best_candidate = None if stage3_payload is None else stage3_payload.get("best_candidate")
    gate_verdict = None if gate_payload is None else str(gate_payload.get("verdict"))
    final_status = "remains thaw-qualified but not canonical"
    stop_stage = None
    if not stage2_payload.get("advancing_candidates"):
        stop_stage = "Stage 2"
    elif not best_candidate:
        stop_stage = "Stage 3"
    elif stage4_payload is None or not stage4_payload.get("stage4_pass"):
        stop_stage = "Stage 4"
    elif stage5_payload is None or not stage5_payload.get("stage5_pass"):
        stop_stage = "Stage 5"
    elif gate_payload is None:
        stop_stage = "Stage 6"
    else:
        best_summary = next(row for row in stage3_payload.get("candidate_summaries", []) if str(row["candidate"]) == str(best_candidate))
        hard_block_fixed = _float(best_summary["candidate_minus_token"]) >= 0.0 and any(
            str(row["lane"]) == "post_pass_b" and _float(row["candidate_minus_token"]) >= 0.0 for row in best_summary.get("block_summaries", [])
        )
        single_ok = _float(best_summary["candidate_minus_single"]) >= -0.02
        successor_pack_path = output.parent / "canonization_successor_candidate_pack.json"
        comparison = {}
        if successor_pack_path.exists():
            comparison = _read_json(successor_pack_path).get("comparison_vs_current", {})
        evidence_stronger = _float(comparison.get("retry_block_sare_delta")) >= 0.0 and _float(comparison.get("combined_sare_delta")) >= 0.0
        if gate_verdict == "PASS: thaw consideration allowed" and hard_block_fixed and single_ok and stage4_payload.get("stage4_pass") and stage5_payload.get("stage5_pass") and evidence_stronger:
            final_status = "qualified for canonization within DoorKey"
        elif gate_verdict != "PASS: thaw consideration allowed":
            final_status = "remains thaw-qualified but not canonical"
            stop_stage = "Stage 6"

    lines = [
        "# Canonization Decision Memo",
        "",
        f"- stop stage: `{stop_stage or 'none'}`",
        f"- final status: `{final_status}`",
        f"- gate verdict: `{gate_verdict or 'not run'}`",
        "",
        "## Stage Summary",
        "",
        f"- Stage 2 advancing candidates: `{stage2_payload.get('advancing_candidates', [])}`",
        f"- Stage 3 best candidate: `{best_candidate}`",
        f"- Stage 4 anti-regression pass: `{None if stage4_payload is None else stage4_payload.get('stage4_pass')}`",
        f"- Stage 5 route/stability pass: `{None if stage5_payload is None else stage5_payload.get('stage5_pass')}`",
        "",
        "## Final Result",
        "",
    ]
    if final_status == "qualified for canonization within DoorKey":
        lines.append("- The hard-block canonization campaign fixes the remaining `post_pass_b`-style weakness, preserves the stronger DoorKey blocks, stays meaningfully routed, and clears the frozen-pack gate strongly enough to recommend canonization within DoorKey only.")
    elif gate_verdict == "PASS: thaw consideration allowed":
        lines.append("- The campaign keeps the candidate lineage above the frozen benchmark gate, but it does not close the remaining canonization blocker strongly enough to replace the frozen benchmark as the canonical DoorKey baseline.")
    else:
        lines.append("- The campaign did not produce a successor strong enough for canonization. The accepted project state remains the existing thaw-qualified-but-not-canonical candidate.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hard-block DoorKey canonization campaign.")
    sub = parser.add_subparsers(dest="command", required=True)

    registration = sub.add_parser("registration")
    registration.add_argument("--campaign-config", required=True)
    registration.add_argument("--output", required=True)

    baseline = sub.add_parser("baseline-sync")
    baseline.add_argument("--campaign-config", required=True)
    baseline.add_argument("--output", required=True)

    shortlist = sub.add_parser("mechanism-shortlist")
    shortlist.add_argument("--campaign-config", required=True)
    shortlist.add_argument("--output", required=True)

    stage2 = sub.add_parser("stage2-report")
    stage2.add_argument("--campaign-config", required=True)
    stage2.add_argument("--stage2-root", required=True)
    stage2.add_argument("--output", required=True)
    stage2.add_argument("--csv", required=True)
    stage2.add_argument("--json", required=True)

    stage3 = sub.add_parser("stage3-report")
    stage3.add_argument("--stage2-json", required=True)
    stage3.add_argument("--stage3-root", required=True)
    stage3.add_argument("--output", required=True)
    stage3.add_argument("--csv", required=True)
    stage3.add_argument("--json", required=True)

    stage4 = sub.add_parser("stage4-report")
    stage4.add_argument("--campaign-config", required=True)
    stage4.add_argument("--stage3-json", required=True)
    stage4.add_argument("--stage4-root", required=True)
    stage4.add_argument("--output", required=True)
    stage4.add_argument("--csv", required=True)
    stage4.add_argument("--json", required=True)

    stage5 = sub.add_parser("stage5-report")
    stage5.add_argument("--campaign-config", required=True)
    stage5.add_argument("--stage3-json", required=True)
    stage5.add_argument("--stage4-json", required=True)
    stage5.add_argument("--route-csv", required=True)
    stage5.add_argument("--device", required=True)
    stage5.add_argument("--episodes", type=int, required=True)
    stage5.add_argument("--output", required=True)
    stage5.add_argument("--csv", required=True)
    stage5.add_argument("--json", required=True)

    stage6 = sub.add_parser("successor-pack")
    stage6.add_argument("--campaign-config", required=True)
    stage6.add_argument("--stage3-json", required=True)
    stage6.add_argument("--stage4-json", required=True)
    stage6.add_argument("--stage5-json", required=True)
    stage6.add_argument("--stage6-root", required=True)
    stage6.add_argument("--summary-output", required=True)
    stage6.add_argument("--metrics-output", required=True)
    stage6.add_argument("--combined-report-output", required=True)
    stage6.add_argument("--combined-report-csv", required=True)
    stage6.add_argument("--retry-report-output", required=True)
    stage6.add_argument("--retry-report-csv", required=True)
    stage6.add_argument("--successor-pack-markdown", required=True)
    stage6.add_argument("--successor-pack-json", required=True)

    decision = sub.add_parser("decision-memo")
    decision.add_argument("--stage2-json", required=True)
    decision.add_argument("--stage3-json")
    decision.add_argument("--stage4-json")
    decision.add_argument("--stage5-json")
    decision.add_argument("--gate-json")
    decision.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "registration":
        _render_registration(_load_yaml(Path(args.campaign_config)), Path(args.output))
        return
    if args.command == "baseline-sync":
        _render_baseline_sync(_load_yaml(Path(args.campaign_config)), Path(args.output))
        return
    if args.command == "mechanism-shortlist":
        _render_mechanism_shortlist(_load_yaml(Path(args.campaign_config)), Path(args.output))
        return
    if args.command == "stage2-report":
        _render_stage2(_load_yaml(Path(args.campaign_config)), _discover_runs(Path(args.stage2_root)), Path(args.output), Path(args.csv), Path(args.json))
        return
    if args.command == "stage3-report":
        _render_stage3(_read_json(Path(args.stage2_json)), _discover_runs(Path(args.stage3_root)), Path(args.output), Path(args.csv), Path(args.json))
        return
    if args.command == "stage4-report":
        _render_stage4(_load_yaml(Path(args.campaign_config)), _read_json(Path(args.stage3_json)), _discover_runs(Path(args.stage4_root)), Path(args.output), Path(args.csv), Path(args.json))
        return
    if args.command == "stage5-report":
        route_rows = _read_csv_rows(Path(args.route_csv)) if Path(args.route_csv).exists() else []
        _render_stage5(
            _load_yaml(Path(args.campaign_config)),
            _read_json(Path(args.stage3_json)),
            _read_json(Path(args.stage4_json)),
            route_rows,
            args.device,
            int(args.episodes),
            Path(args.output),
            Path(args.csv),
            Path(args.json),
        )
        return
    if args.command == "successor-pack":
        campaign = _load_yaml(Path(args.campaign_config))
        stage3_payload = _read_json(Path(args.stage3_json))
        stage4_payload = _read_json(Path(args.stage4_json))
        stage5_payload = _read_json(Path(args.stage5_json))
        retry_records = _discover_runs(Path(args.stage6_root))
        _render_successor_pack(
            campaign,
            stage3_payload,
            stage4_payload,
            stage5_payload,
            retry_records,
            Path(args.summary_output),
            Path(args.metrics_output),
            Path(args.combined_report_output),
            Path(args.combined_report_csv),
            Path(args.retry_report_output),
            Path(args.retry_report_csv),
            Path(args.successor_pack_markdown),
            Path(args.successor_pack_json),
        )
        return
    if args.command == "decision-memo":
        stage2_payload = _read_json(Path(args.stage2_json))
        stage3_payload = _read_json(Path(args.stage3_json)) if args.stage3_json else None
        stage4_payload = _read_json(Path(args.stage4_json)) if args.stage4_json else None
        stage5_payload = _read_json(Path(args.stage5_json)) if args.stage5_json else None
        gate_payload = _read_json(Path(args.gate_json)) if args.gate_json else None
        _render_decision_memo(stage2_payload, stage3_payload, stage4_payload, stage5_payload, gate_payload, Path(args.output))
        return


if __name__ == "__main__":
    main()
