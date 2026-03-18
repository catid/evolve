from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psmn_rl.analysis.benchmark_pack import sha256_path
from psmn_rl.analysis.claim_gate import evaluate_pack_claim_gate_from_paths, render_pack_gate_report
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


def _block_specs(campaign: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [dict(item) for item in campaign["seed_groups"][key]["blocks"]]


def _block_lanes(campaign: dict[str, Any], key: str) -> tuple[str, ...]:
    return tuple(str(item["lane"]) for item in _block_specs(campaign, key))


def _lane_seed_pairs(campaign: dict[str, Any], key: str) -> list[tuple[str, int]]:
    return _lane_seed_list(campaign["seed_groups"][key])


def _family_stats(rows: list[dict[str, Any]], candidate: str, label: str, lanes: tuple[str, ...]) -> dict[str, float]:
    values = [
        _float(row["final_greedy_success"])
        for row in rows
        if str(row["candidate"]) == candidate and str(row["label"]) == label and str(row["lane"]) in lanes
    ]
    return _stats(values)


def _block_summary_from_rows(rows: list[dict[str, Any]], candidate: str, lanes: tuple[str, ...]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for lane in lanes:
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


def _load_current_rows(campaign: dict[str, Any], lanes: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane in lanes:
        root = Path(campaign["current_candidate_roots"][lane])
        for summary_path in sorted(root.glob("seed_*/kl_lss_*/summary.json")):
            run_dir = summary_path.parent
            seed_dir = run_dir.parent
            if not seed_dir.name.startswith("seed_"):
                continue
            rows.append(
                {
                    "candidate": str(campaign["current_candidate_name"]),
                    "lane": lane,
                    "seed": int(seed_dir.name.split("_", 1)[1]),
                    "label": run_dir.name,
                    "run_dir": str(run_dir),
                    "summary": _read_json(summary_path),
                }
            )
    return [
        {
            **row,
            **{
                "final_greedy_success": _float(row["summary"].get("final_greedy_success")),
                "best_round_index": _int(row["summary"].get("best_round_index")),
                "best_round_greedy_success": _float(row["summary"].get("best_round_greedy_success")),
            },
        }
        for row in rows
    ]


def _current_candidate_run_dir(campaign: dict[str, Any], lane: str, seed: int) -> Path:
    return Path(campaign["current_candidate_roots"][lane]) / f"seed_{seed}" / "kl_lss_sare"


def _candidate_stage3_root(campaign: dict[str, Any], candidate: str) -> Path:
    meta = campaign["candidates"][candidate]
    reuse = meta.get("reuse_root")
    if reuse:
        return Path(str(reuse))
    return Path(campaign["stage_roots"]["stage3_fairness"]) / candidate


def _candidate_run_dir(campaign: dict[str, Any], candidate: str, lane: str, seed: int) -> Path:
    dev_lanes = set(_block_lanes(campaign, "hard_family_dev"))
    holdout_lanes = set(_block_lanes(campaign, "hard_family_holdout"))
    healthy_lanes = set(_block_lanes(campaign, "healthy_blocks"))
    if lane in dev_lanes:
        return _candidate_stage3_root(campaign, candidate) / lane / f"seed_{seed}" / "kl_lss_sare"
    if lane in holdout_lanes:
        return Path(campaign["stage_roots"]["stage4_holdout"]) / candidate / lane / f"seed_{seed}" / "kl_lss_sare"
    if lane in healthy_lanes:
        return Path(campaign["stage_roots"]["stage5_antiregression"]) / candidate / lane / f"seed_{seed}" / "kl_lss_sare"
    raise KeyError(f"Unknown lane: {lane}")


def _summary_compare_row(campaign: dict[str, Any], candidate: str, lane: str, seed: int) -> dict[str, Any]:
    candidate_summary = _read_json(_candidate_run_dir(campaign, candidate, lane, seed) / "summary.json")
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


def _candidate_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "lane": str(row["lane"]),
        "seed": int(row["seed"]),
        "label": str(row["label"]),
        "eval_success_rate": _float(row["final_greedy_success"]),
        "run_dir": str(row["run_dir"]),
    }


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


def _render_definition(campaign: dict[str, Any], output: Path) -> None:
    dev_lanes = _block_lanes(campaign, "hard_family_dev")
    holdout_lanes = _block_lanes(campaign, "hard_family_holdout")
    current_rows = _load_current_rows(campaign, dev_lanes + holdout_lanes)
    lines = [
        "# Hard-Family Definition",
        "",
        f"- current candidate: `{campaign['current_candidate_name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Family Definition",
        "",
        "- The hard family is defined by a shared late-phase DoorKey failure signature: the thaw-qualified routed candidate remains behind matched KL learner-state `token_dense` under the external 64-episode path, the gap is concentrated after unlock, and the block is hard without broadening beyond DoorKey.",
        "- Development blocks stay on the explicit post-pass family introduced in the canonization lane: `post_pass_b` and `post_pass_c`.",
        "- The holdout stays on an independent previously known hard block: `fresh_final`. It is withheld from intervention selection in this program and used only as the hard-family test split.",
        "",
        "## Current Candidate Snapshot",
        "",
        "| Block | Seed | Variant | Greedy Success |",
        "| --- | --- | --- | ---: |",
    ]
    for row in sorted(current_rows, key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"]))):
        lines.append(
            f"| {row['lane']} | {row['seed']} | {DISPLAY_NAMES[str(row['label'])]} | {_format_float(_float(row['final_greedy_success']))} |"
        )
    lines.extend(
        [
            "",
            "## Family Membership Notes",
            "",
            "- `post_pass_b`: retained as the existing canonization blocker because matched `token_dense` is still perfect while the current candidate stays below it.",
            "- `post_pass_c`: kept in development because it was introduced as the same post-pass prime-seed family and shares the late cleanup imbalance targeted by the hard-block fixes.",
            "- `fresh_final`: used as holdout because it is an independent historical hard DoorKey block where matched `token_dense` still dominates the current candidate and the late-phase routed failure story remains relevant.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_registration(campaign: dict[str, Any], output: Path) -> None:
    lines = [
        "# Hard-Family Campaign Registration",
        "",
        f"- frozen benchmark pack: `{campaign['frozen_pack']}`",
        f"- current thaw-qualified candidate pack: `{campaign['current_candidate_pack']}`",
        f"- current candidate: `{campaign['current_candidate_name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Splits",
        "",
        f"- dev blocks: `{_block_specs(campaign, 'hard_family_dev')}`",
        f"- holdout blocks: `{_block_specs(campaign, 'hard_family_holdout')}`",
        f"- anti-regression healthy blocks: `{_block_specs(campaign, 'healthy_blocks')}`",
        "",
        "## Stage Gates",
        "",
        "- Stage 2: the candidate must materially improve over the thaw-qualified incumbent on the hard-family dev split, narrow the matched token_dense gap, and avoid new complete-seed failures.",
        "- Stage 3: on the hard-family dev split, `SARE` must no longer trail matched `token_dense` and must remain at least competitive with matched `single_expert`.",
        "- Stage 4: the best dev survivor must generalize that gain to the hard-family holdout without falling back to the original token-dense-led pattern.",
        "- Stage 5: the holdout survivor must not materially worsen the healthier DoorKey blocks or lose thaw-qualified status.",
        "- Stage 6: route probes on dev, holdout, and healthy cases must still show meaningful routing dependence.",
        "- Stage 7: the improved candidate must avoid narrow checkpoint spikes on dev, holdout, and healthy cases.",
        "- Stage 8: the successor candidate pack must validate, clear the pack-based gate relative to the frozen benchmark, and only then decide canonical vs thaw-qualified vs fallback.",
        "",
        "## Reports",
        "",
        f"- family definition: `{campaign['reports']['definition']}`",
        f"- baseline sync: `{campaign['reports']['baseline_sync']}`",
        f"- shortlist: `{campaign['reports']['stage1_shortlist']}`",
        f"- dev screening: `{campaign['reports']['stage2_report']}`",
        f"- fairness: `{campaign['reports']['stage3_report']}`",
        f"- holdout: `{campaign['reports']['stage4_report']}`",
        f"- anti-regression: `{campaign['reports']['stage5_report']}`",
        f"- route validation: `{campaign['reports']['stage6_report']}`",
        f"- stability: `{campaign['reports']['stage7_report']}`",
        f"- successor pack: `{campaign['reports']['successor_pack_json']}`",
        f"- decision memo: `{campaign['reports']['decision_memo']}`",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_baseline_sync(campaign: dict[str, Any], output: Path) -> None:
    manifest = _load_yaml(Path(campaign["frozen_manifest"]))
    current_pack = _read_json(Path(campaign["current_candidate_pack"]))
    dev_lanes = _block_lanes(campaign, "hard_family_dev")
    holdout_lanes = _block_lanes(campaign, "hard_family_holdout")
    current_rows = _load_current_rows(campaign, dev_lanes + holdout_lanes)
    dev_family = _family_stats(current_rows, str(campaign["current_candidate_name"]), "kl_lss_sare", dev_lanes)
    dev_token = _family_stats(current_rows, str(campaign["current_candidate_name"]), "kl_lss_token_dense", dev_lanes)
    holdout_family = _family_stats(current_rows, str(campaign["current_candidate_name"]), "kl_lss_sare", holdout_lanes)
    holdout_token = _family_stats(current_rows, str(campaign["current_candidate_name"]), "kl_lss_token_dense", holdout_lanes)
    lines = [
        "# Hard-Family Baseline Sync",
        "",
        f"- frozen benchmark pack: `{campaign['frozen_pack']}`",
        f"- current thaw-qualified candidate pack: `{campaign['current_candidate_pack']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Frozen Gate Thresholds",
        "",
        f"- retry-block KL learner-state `SARE` mean to beat: `{manifest['thresholds']['retry_block_means']['kl_lss_sare']:.4f}`",
        f"- retry-block KL learner-state `single_expert` mean to match or beat: `{manifest['thresholds']['retry_block_means']['kl_lss_single_expert']:.4f}`",
        f"- combined KL learner-state `SARE` mean to preserve: `{manifest['thresholds']['combined_means']['kl_lss_sare']:.4f}`",
        "",
        "## Current Candidate Snapshot",
        "",
        f"- retry-block KL learner-state `SARE` mean: `{_float(current_pack['metrics']['retry_block']['kl_lss_sare']['mean']):.4f}`",
        f"- retry-block KL learner-state `single_expert` mean: `{_float(current_pack['metrics']['retry_block']['kl_lss_single_expert']['mean']):.4f}`",
        f"- frozen-comparable combined KL learner-state `SARE` mean: `{_float(current_pack['metrics']['combined']['kl_lss_sare']['mean']):.4f}`",
        f"- dev-family KL learner-state `SARE` mean: `{dev_family['mean']:.4f}` vs matched token_dense `{dev_token['mean']:.4f}`",
        f"- holdout KL learner-state `SARE` mean: `{holdout_family['mean']:.4f}` vs matched token_dense `{holdout_token['mean']:.4f}`",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_shortlist(campaign: dict[str, Any], output: Path) -> None:
    lines = [
        "# Hard-Family Stage 1 Shortlist",
        "",
        "| Candidate | Intervention Family | Mechanism Hypothesis | Hard-Family Target | Broader DoorKey Risk |",
        "| --- | --- | --- | --- | --- |",
    ]
    for candidate_id, meta in campaign["candidates"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{candidate_id}`",
                    str(meta["intervention_family"]),
                    str(meta["hypothesis"]),
                    str(meta["hard_family_target"]),
                    str(meta["broader_risk"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The shortlist stays inside the existing teacher-guided KL learner-state family.",
            "- The new candidates only add phase-balanced recent replay and a modest disagreement bonus on top of the already strongest round-5 late-phase intervention; this is a bounded hard-family program rather than a broad search.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _discover_stage2_records(campaign: dict[str, Any]) -> list[RunRecord]:
    records: list[RunRecord] = []
    for candidate, meta in campaign["candidates"].items():
        reuse_root = meta.get("reuse_root")
        if reuse_root:
            records.extend(
                [
                    record
                    for record in _discover_runs(Path(str(reuse_root)).parent)
                    if record.candidate == candidate and record.lane in _block_lanes(campaign, "hard_family_dev")
                ]
            )
        else:
            records.extend(
                [
                    record
                    for record in _discover_runs(Path(campaign["stage_roots"]["stage2_dev"]))
                    if record.candidate == candidate and record.lane in _block_lanes(campaign, "hard_family_dev")
                ]
            )
    return records


def _stage2_pass(summary: dict[str, Any], incumbent_failures: int) -> bool:
    return (
        summary["candidate_mean"] >= summary["incumbent_mean"] + 0.02
        and summary["gap_narrowing_vs_token"] >= 0.05
        and int(summary["candidate_failures"]) <= incumbent_failures
        and _float(summary["max_block_delta"]) >= 0.05
    )


def _render_stage2(campaign: dict[str, Any], output: Path, csv_output: Path, json_output: Path) -> None:
    dev_lanes = _block_lanes(campaign, "hard_family_dev")
    current_rows = _load_current_rows(campaign, dev_lanes)
    detail_rows = [_candidate_row(record) for record in _discover_stage2_records(campaign)]
    incumbent = str(campaign["current_candidate_name"])
    control_lookup = {
        (str(row["lane"]), int(row["seed"]), str(row["label"])): row
        for row in current_rows
        if str(row["label"]) in CURRENT_CONTROLS
    }
    screening_rows: list[dict[str, Any]] = []
    for row in sorted(
        [item for item in detail_rows if str(item["label"]) == "kl_lss_sare"],
        key=lambda item: (str(item["candidate"]), str(item["lane"]), int(item["seed"])),
    ):
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
    incumbent_family = _family_stats(current_rows, incumbent, "kl_lss_sare", dev_lanes)
    token_family = _family_stats(current_rows, incumbent, "kl_lss_token_dense", dev_lanes)
    single_family = _family_stats(current_rows, incumbent, "kl_lss_single_expert", dev_lanes)
    incumbent_failures = int(incumbent_family["complete_seed_failures"])
    candidate_summaries: list[dict[str, Any]] = []
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in screening_rows:
        by_candidate.setdefault(str(row["candidate"]), []).append(row)
    for candidate, rows in sorted(by_candidate.items()):
        candidate_stats = _stats([_float(row["final_greedy_success"]) for row in rows])
        block_summaries: list[dict[str, Any]] = []
        block_deltas: list[float] = []
        for lane in dev_lanes:
            lane_rows = [row for row in rows if str(row["lane"]) == lane]
            lane_mean = _stats([_float(row["final_greedy_success"]) for row in lane_rows])["mean"]
            current_mean = _stats(
                [
                    _float(row["final_greedy_success"])
                    for row in current_rows
                    if str(row["lane"]) == lane and str(row["label"]) == "kl_lss_sare"
                ]
            )["mean"]
            block_summaries.append({"lane": lane, "candidate_mean": lane_mean, "delta_vs_current": lane_mean - current_mean})
            block_deltas.append(lane_mean - current_mean)
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
            "max_block_delta": max(block_deltas) if block_deltas else 0.0,
            "block_summaries": block_summaries,
        }
        summary["stage2_pass"] = _stage2_pass(summary, incumbent_failures)
        candidate_summaries.append(summary)
    advancing = [
        row["candidate"]
        for row in sorted(
            [row for row in candidate_summaries if row["stage2_pass"]],
            key=lambda item: (item["candidate_mean"], item["gap_narrowing_vs_token"]),
            reverse=True,
        )[:2]
    ]

    lines = [
        "# Hard-Family Stage 2 Dev Screening",
        "",
        f"- current dev-family KL learner-state `SARE` mean: `{incumbent_family['mean']:.4f}`",
        f"- current dev-family matched token_dense mean: `{token_family['mean']:.4f}`",
        f"- current dev-family matched single_expert mean: `{single_family['mean']:.4f}`",
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
            "| Candidate | Dev Mean | Δ vs Current SARE | Gap Narrowing vs token_dense | Candidate - token_dense | Candidate - single_expert | Complete-Seed Failures | Max Block Δ vs Current | Stage 2 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
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
                    _format_float(row["max_block_delta"]),
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
            f"- advancing candidates: `{advancing}`",
            "- Stage 2 keeps only candidates that beat the thaw-qualified incumbent on the dev split, narrow the matched token_dense gap, and avoid new complete-seed failures.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(csv_output, screening_rows)
    _write_json(
        json_output,
        {
            "stage": "stage2",
            "current_family": {
                "sare_mean": incumbent_family["mean"],
                "token_mean": token_family["mean"],
                "single_mean": single_family["mean"],
                "candidate_failures": incumbent_failures,
            },
            "detail_rows": screening_rows,
            "candidate_summaries": candidate_summaries,
            "advancing_candidates": advancing,
        },
    )


def _discover_stage3_records(campaign: dict[str, Any], candidates: list[str]) -> list[RunRecord]:
    records: list[RunRecord] = []
    for candidate in candidates:
        root = _candidate_stage3_root(campaign, candidate)
        records.extend(
            [
                record
                for record in _discover_runs(root.parent if root.name == candidate else root)
                if record.candidate == candidate and record.lane in _block_lanes(campaign, "hard_family_dev")
            ]
        )
    return records


def _stage3_pass(block_summaries: list[dict[str, Any]], family_summary: dict[str, Any]) -> bool:
    block_single_ok = all(_float(row["candidate_minus_single"]) >= -0.05 for row in block_summaries)
    return (
        _float(family_summary["candidate_minus_token"]) >= 0.0
        and _float(family_summary["candidate_minus_single"]) >= -0.02
        and int(family_summary["candidate_failures"]) == 0
        and block_single_ok
    )


def _select_best_case(stage_rows: list[dict[str, Any]], candidate: str) -> dict[str, Any] | None:
    rows = [row for row in stage_rows if str(row["candidate"]) == candidate and str(row["label"]) == "kl_lss_sare"]
    if not rows:
        return None
    selected = max(rows, key=lambda item: (_float(item["final_greedy_success"]), -_float(item["best_round_disagreement"])))
    return {"lane": str(selected["lane"]), "seed": int(selected["seed"])}


def _render_stage3(campaign: dict[str, Any], stage2_payload: dict[str, Any], output: Path, csv_output: Path, json_output: Path) -> None:
    dev_lanes = _block_lanes(campaign, "hard_family_dev")
    detail_rows = [_candidate_row(record) for record in _discover_stage3_records(campaign, stage2_payload.get("advancing_candidates", []))]
    candidate_summaries: list[dict[str, Any]] = []
    for candidate in stage2_payload.get("advancing_candidates", []):
        candidate_rows = [row for row in detail_rows if str(row["candidate"]) == candidate]
        block_summaries = _block_summary_from_rows(candidate_rows, candidate, dev_lanes)
        family_sare = _family_stats(candidate_rows, candidate, "kl_lss_sare", dev_lanes)
        family_token = _family_stats(candidate_rows, candidate, "kl_lss_token_dense", dev_lanes)
        family_single = _family_stats(candidate_rows, candidate, "kl_lss_single_expert", dev_lanes)
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
    selected_dev_case = None
    if survivors:
        best_summary = max([row for row in candidate_summaries if row["stage3_pass"]], key=lambda item: (item["candidate_mean"], item["candidate_minus_token"]))
        best_candidate = str(best_summary["candidate"])
        selected_dev_case = _select_best_case(detail_rows, best_candidate)
    lines = [
        "# Hard-Family Stage 3 Fairness",
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
            "| Candidate | Dev SARE | Dev token_dense | Dev single_expert | Dev SARE-token | Dev SARE-single | Complete-Seed Failures | Stage 3 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(candidate_summaries, key=lambda item: item["candidate_mean"], reverse=True):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['candidate']}`",
                    _format_float(row["candidate_mean"]),
                    _format_float(row["token_mean"]),
                    _format_float(row["single_mean"]),
                    _format_float(row["candidate_minus_token"]),
                    _format_float(row["candidate_minus_single"]),
                    str(int(row["candidate_failures"])),
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
            f"- selected dev case for route validation: `{selected_dev_case}`",
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
            "selected_dev_case": selected_dev_case,
        },
    )


def _discover_stage4_records(root: Path, best_candidate: str) -> list[RunRecord]:
    return [record for record in _discover_runs(root) if record.candidate == best_candidate]


def _render_stage4(campaign: dict[str, Any], stage3_payload: dict[str, Any], output: Path, csv_output: Path, json_output: Path) -> None:
    holdout_lanes = _block_lanes(campaign, "hard_family_holdout")
    best_candidate = stage3_payload.get("best_candidate")
    if not best_candidate:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("# Hard-Family Stage 4 Holdout\n\n- skipped: `no candidate survived dev fairness`\n", encoding="utf-8")
        _write_csv(csv_output, [])
        _write_json(
            json_output,
            {
                "stage": "stage4",
                "best_candidate": None,
                "detail_rows": [],
                "holdout_summary": None,
                "stage4_pass": False,
                "selected_holdout_case": None,
                "skipped": True,
            },
        )
        return
    detail_rows = [_candidate_row(record) for record in _discover_stage4_records(Path(campaign["stage_roots"]["stage4_holdout"]), str(best_candidate))]
    current_rows = _load_current_rows(campaign, holdout_lanes)
    holdout_sare = _family_stats(detail_rows, str(best_candidate), "kl_lss_sare", holdout_lanes)
    holdout_token = _family_stats(detail_rows, str(best_candidate), "kl_lss_token_dense", holdout_lanes)
    holdout_single = _family_stats(detail_rows, str(best_candidate), "kl_lss_single_expert", holdout_lanes)
    current_holdout = _family_stats(current_rows, str(campaign["current_candidate_name"]), "kl_lss_sare", holdout_lanes)
    holdout_summary = {
        "candidate": str(best_candidate),
        "candidate_mean": holdout_sare["mean"],
        "token_mean": holdout_token["mean"],
        "single_mean": holdout_single["mean"],
        "candidate_minus_token": holdout_sare["mean"] - holdout_token["mean"],
        "candidate_minus_single": holdout_sare["mean"] - holdout_single["mean"],
        "candidate_failures": holdout_sare["complete_seed_failures"],
        "delta_vs_current": holdout_sare["mean"] - current_holdout["mean"],
    }
    stage4_pass = (
        _float(holdout_summary["candidate_minus_token"]) >= 0.0
        and _float(holdout_summary["candidate_minus_single"]) >= -0.02
        and int(holdout_summary["candidate_failures"]) == 0
    )
    selected_holdout_case = _select_best_case(detail_rows, str(best_candidate))
    lines = [
        "# Hard-Family Stage 4 Holdout",
        "",
        f"- best candidate: `{best_candidate}`",
        f"- current holdout KL learner-state `SARE` mean: `{current_holdout['mean']:.4f}`",
        f"- candidate holdout KL learner-state `SARE` mean: `{holdout_sare['mean']:.4f}`",
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
            "## Holdout Summary",
            "",
            f"- candidate holdout SARE-token delta: `{holdout_summary['candidate_minus_token']:.4f}`",
            f"- candidate holdout SARE-single delta: `{holdout_summary['candidate_minus_single']:.4f}`",
            f"- candidate holdout delta vs current thaw-qualified candidate: `{holdout_summary['delta_vs_current']:.4f}`",
            f"- stage-4 status: `{'pass' if stage4_pass else 'stop'}`",
            f"- selected holdout case for route validation: `{selected_holdout_case}`",
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
            "holdout_summary": holdout_summary,
            "selected_holdout_case": selected_holdout_case,
            "stage4_pass": stage4_pass,
        },
    )


def _healthy_block_summary(rows: list[dict[str, Any]], lanes: tuple[str, ...]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for lane in lanes:
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


def _render_stage5(campaign: dict[str, Any], stage3_payload: dict[str, Any], stage4_payload: dict[str, Any], output: Path, csv_output: Path, json_output: Path) -> None:
    healthy_lanes = _block_lanes(campaign, "healthy_blocks")
    best_candidate = stage3_payload.get("best_candidate")
    if not best_candidate or not stage4_payload.get("stage4_pass"):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("# Hard-Family Stage 5 Anti-Regression\n\n- skipped: `candidate did not survive holdout evaluation`\n", encoding="utf-8")
        _write_csv(csv_output, [])
        _write_json(
            json_output,
            {
                "stage": "stage5",
                "best_candidate": best_candidate,
                "detail_rows": [],
                "block_summaries": [],
                "stage5_pass": False,
                "selected_healthy_case": None,
                "skipped": True,
            },
        )
        return
    detail_rows = [
        _candidate_row(record)
        for record in _discover_runs(Path(campaign["stage_roots"]["stage5_antiregression"]))
        if record.candidate == str(best_candidate) and record.lane in healthy_lanes
    ]
    current_rows = _load_current_rows(campaign, healthy_lanes)
    current_sare = _family_stats(current_rows, str(campaign["current_candidate_name"]), "kl_lss_sare", healthy_lanes)
    candidate_sare = _family_stats(detail_rows, str(best_candidate), "kl_lss_sare", healthy_lanes)
    current_lookup = {
        (str(row["lane"]), int(row["seed"]), str(row["label"])): row for row in current_rows if str(row["label"]) == "kl_lss_sare"
    }
    healthy_failures: list[tuple[str, int]] = []
    for row in detail_rows:
        if str(row["label"]) != "kl_lss_sare":
            continue
        current_row = current_lookup[(str(row["lane"]), int(row["seed"]), "kl_lss_sare")]
        if _float(current_row["final_greedy_success"]) > 0.0 and _float(row["final_greedy_success"]) <= 0.0:
            healthy_failures.append((str(row["lane"]), int(row["seed"])))
    stage5_pass = (
        candidate_sare["mean"] >= current_sare["mean"] - 0.02
        and candidate_sare["complete_seed_failures"] <= current_sare["complete_seed_failures"]
        and not healthy_failures
    )
    block_summaries = _healthy_block_summary(detail_rows, healthy_lanes)
    selected_healthy_case = _select_best_case(detail_rows, str(best_candidate))
    lines = [
        "# Hard-Family Stage 5 Anti-Regression",
        "",
        f"- best candidate: `{best_candidate}`",
        f"- current healthy-block KL learner-state `SARE` mean: `{current_sare['mean']:.4f}`",
        f"- candidate healthy-block KL learner-state `SARE` mean: `{candidate_sare['mean']:.4f}`",
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
            f"| {row['lane']} | {_format_float(row['candidate_mean'])} | {_format_float(row['token_mean'])} | {_format_float(row['single_mean'])} | {_format_float(row['candidate_minus_token'])} | {_format_float(row['candidate_minus_single'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- new complete-seed failures on previously healthy blocks: `{healthy_failures}`",
            f"- selected healthy case for route validation: `{selected_healthy_case}`",
            f"- stage-5 status: `{'pass' if stage5_pass else 'stop'}`",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(csv_output, detail_rows)
    _write_json(
        json_output,
        {
            "stage": "stage5",
            "best_candidate": best_candidate,
            "detail_rows": detail_rows,
            "block_summaries": block_summaries,
            "current_healthy_sare_mean": current_sare["mean"],
            "candidate_healthy_sare_mean": candidate_sare["mean"],
            "healthy_failures": healthy_failures,
            "selected_healthy_case": selected_healthy_case,
            "stage5_pass": stage5_pass,
        },
    )


def _stability_targets(campaign: dict[str, Any], candidate: str, cases: list[dict[str, Any]]) -> list[EvalTarget]:
    targets: list[EvalTarget] = []
    for case in cases:
        lane = str(case["lane"])
        seed = int(case["seed"])
        for source in ("candidate", "baseline"):
            run_dir = _candidate_run_dir(campaign, candidate, lane, seed) if source == "candidate" else _current_candidate_run_dir(campaign, lane, seed)
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
                        stage="hard_family_stage7",
                        round_index=round_index,
                        metadata={"lane": lane, "source": source, "case_kind": str(case["kind"])},
                        command_path=run_dir / "command.txt",
                    )
                )
    return targets


def _render_stage6(campaign: dict[str, Any], stage3_payload: dict[str, Any], stage4_payload: dict[str, Any], stage5_payload: dict[str, Any], route_rows: list[dict[str, Any]], output: Path, csv_output: Path, json_output: Path) -> None:
    best_candidate = stage3_payload.get("best_candidate")
    selected_dev = stage3_payload.get("selected_dev_case")
    selected_holdout = stage4_payload.get("selected_holdout_case")
    selected_healthy = stage5_payload.get("selected_healthy_case")
    if not best_candidate or not stage5_payload.get("stage5_pass") or not selected_dev or not selected_holdout or not selected_healthy:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("# Hard-Family Stage 6 Route Validation\n\n- skipped: `candidate did not survive anti-regression`\n", encoding="utf-8")
        _write_csv(csv_output, [])
        _write_json(
            json_output,
            {
                "stage": "stage6",
                "best_candidate": best_candidate,
                "route_summaries": [],
                "stage6_pass": False,
                "skipped": True,
            },
        )
        return
    route_grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in route_rows:
        route_grouped.setdefault((str(row["lane"]), int(row["seed"])), []).append(row)
    cases = [
        {"kind": "dev", **selected_dev},
        {"kind": "holdout", **selected_holdout},
        {"kind": "healthy", **selected_healthy},
    ]
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
                "case_kind": str(case["kind"]),
                "lane": lane,
                "seed": seed,
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
    fixed_ok = bool(route_summaries) and all(row["fixed_drop"] >= 0.25 for row in route_summaries)
    ablation_ok = bool(route_summaries) and all(row["worst_ablation_drop"] >= 0.25 for row in route_summaries)
    random_ok = sum(1 for row in route_summaries if row["random_drop"] >= 0.25) >= 2
    stage6_pass = fixed_ok and ablation_ok and random_ok
    lines = [
        "# Hard-Family Stage 6 Route Validation",
        "",
        f"- best candidate: `{best_candidate}`",
        "",
        "| Case | Lane | Seed | Baseline | Fixed-Router Drop | Route-Randomization Drop | Worst Ablation Drop | Final Disagreement Δ | Final Post-Unlock Frac Δ | Final Route Entropy Δ | Final Path Entropy Δ |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
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
            f"- stage-6 status: `{'pass' if stage6_pass else 'stop'}`",
            "- The route summary compares the best hard-family candidate against the current thaw-qualified incumbent on the same dev, holdout, and healthy seeds so a hard-family gain cannot hide by erasing route structure.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(csv_output, route_summaries)
    _write_json(
        json_output,
        {
            "stage": "stage6",
            "best_candidate": best_candidate,
            "route_summaries": route_summaries,
            "stage6_pass": stage6_pass,
        },
    )


def _render_stage7(campaign: dict[str, Any], stage3_payload: dict[str, Any], stage4_payload: dict[str, Any], stage5_payload: dict[str, Any], stage6_payload: dict[str, Any], device: str, episodes: int, output: Path, csv_output: Path, json_output: Path) -> None:
    best_candidate = stage3_payload.get("best_candidate")
    selected_dev = stage3_payload.get("selected_dev_case")
    selected_holdout = stage4_payload.get("selected_holdout_case")
    selected_healthy = stage5_payload.get("selected_healthy_case")
    if not best_candidate or not stage6_payload.get("stage6_pass") or not selected_dev or not selected_holdout or not selected_healthy:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("# Hard-Family Stage 7 Stability\n\n- skipped: `candidate did not survive route validation`\n", encoding="utf-8")
        _write_csv(csv_output, [])
        _write_json(
            json_output,
            {
                "stage": "stage7",
                "best_candidate": best_candidate,
                "stability_summaries": [],
                "stage7_pass": False,
                "skipped": True,
            },
        )
        return
    cases = [
        {"kind": "dev", **selected_dev},
        {"kind": "holdout", **selected_holdout},
        {"kind": "healthy", **selected_healthy},
    ]
    eval_rows = [row for row in _evaluate_targets(_stability_targets(campaign, str(best_candidate), cases), device, episodes) if str(row["mode"]) == "greedy"]
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
    candidate_summaries = [row for row in stability_summaries if row["source"] == "candidate"]
    stage7_pass = all(row["classification"] != "narrow_spike" for row in candidate_summaries)
    lines = [
        "# Hard-Family Stage 7 Stability",
        "",
        f"- best candidate: `{best_candidate}`",
        "",
        "| Case | Lane | Seed | Source | Round Successes | Classification |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in stability_summaries:
        values = ", ".join(f"{value:.4f}" for value in row["round_successes"])
        lines.append(f"| {row['case_kind']} | {row['lane']} | {row['seed']} | {row['source']} | `{values}` | `{row['classification']}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- stage-7 status: `{'pass' if stage7_pass else 'stop'}`",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(csv_output, stability_summaries)
    _write_json(
        json_output,
        {
            "stage": "stage7",
            "best_candidate": best_candidate,
            "stability_summaries": stability_summaries,
            "stage7_pass": stage7_pass,
        },
    )


def _render_successor_pack(
    campaign: dict[str, Any],
    stage3_payload: dict[str, Any],
    stage4_payload: dict[str, Any],
    stage5_payload: dict[str, Any],
    stage6_payload: dict[str, Any],
    stage7_payload: dict[str, Any],
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
    stage5_rows = [_candidate_metric_row(row) for row in stage5_payload.get("detail_rows", []) if str(row["label"]) in CURRENT_CONTROLS]
    stage4_rows = [_candidate_metric_row(row) for row in stage4_payload.get("detail_rows", []) if str(row["label"]) in CURRENT_CONTROLS]
    candidate_rows = _dedupe_metric_rows(stage5_rows + stage4_rows)
    combined_lane_seeds = _lane_seed_pairs(campaign, "frozen_combined")
    retry_lane_seeds = _lane_seed_pairs(campaign, "hard_family_holdout")
    combined_rows = _dedupe_metric_rows(_pack_metric_rows(candidate_rows + baseline_rows, combined_lane_seeds))
    retry_rows_pack = _dedupe_metric_rows(_pack_metric_rows(candidate_rows + baseline_rows, retry_lane_seeds))
    combined_metrics = _metric_block(combined_rows)
    retry_metrics = _metric_block(retry_rows_pack)

    combined_lines = [
        "# Hard-Family Candidate Frozen-Comparable Combined Report",
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
        combined_lines.append(f"| {DISPLAY_NAMES[label]} | `{combined_metrics[label]['mean']:.4f}` | `{int(combined_metrics[label]['complete_seed_failures'])}` |")
    combined_md.parent.mkdir(parents=True, exist_ok=True)
    combined_md.write_text("\n".join(combined_lines) + "\n", encoding="utf-8")
    _write_csv(combined_csv, combined_rows)

    retry_lines = [
        "# Hard-Family Candidate Holdout / Retry Report",
        "",
        "| Seed | recovered token_dense | KL learner-state token_dense | KL learner-state single_expert | baseline PPO SARE | KL learner-state SARE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lookup = {(str(row["lane"]), int(row["seed"]), str(row["label"])): row for row in retry_rows_pack}
    for lane, seed in retry_lane_seeds:
        retry_lines.append(
            "| "
            + " | ".join(
                [
                    str(seed),
                    _format_float(lookup[(lane, seed, "recovered_token_dense")]["eval_success_rate"]),
                    _format_float(lookup[(lane, seed, "kl_lss_token_dense")]["eval_success_rate"]),
                    _format_float(lookup[(lane, seed, "kl_lss_single_expert")]["eval_success_rate"]),
                    _format_float(lookup[(lane, seed, "baseline_sare")]["eval_success_rate"]),
                    _format_float(lookup[(lane, seed, "kl_lss_sare")]["eval_success_rate"]),
                ]
            )
            + " |"
        )
    retry_lines.extend(["", "## Summary", "", "| Variant | Mean | Complete-Seed Failures |", "| --- | ---: | ---: |"])
    for label in LABELS:
        retry_lines.append(f"| {DISPLAY_NAMES[label]} | `{retry_metrics[label]['mean']:.4f}` | `{int(retry_metrics[label]['complete_seed_failures'])}` |")
    retry_md.parent.mkdir(parents=True, exist_ok=True)
    retry_md.write_text("\n".join(retry_lines) + "\n", encoding="utf-8")
    _write_csv(retry_csv, retry_rows_pack)

    current_pack = _read_json(Path(campaign["current_candidate_pack"]))
    current_retry = _float(current_pack["metrics"]["retry_block"]["kl_lss_sare"]["mean"])
    current_combined = _float(current_pack["metrics"]["combined"]["kl_lss_sare"]["mean"])
    dev_summary = next(row for row in stage3_payload["candidate_summaries"] if str(row["candidate"]) == best_candidate)
    holdout_summary = stage4_payload["holdout_summary"]
    summary_lines = [
        "# Hard-Family Candidate Summary",
        "",
        f"- candidate: `{best_candidate}`",
        f"- hard-family dev KL learner-state `SARE` mean: `{dev_summary['candidate_mean']:.4f}`",
        f"- hard-family dev matched token_dense mean: `{dev_summary['token_mean']:.4f}`",
        f"- hard-family holdout KL learner-state `SARE` mean: `{holdout_summary['candidate_mean']:.4f}`",
        f"- hard-family holdout matched token_dense mean: `{holdout_summary['token_mean']:.4f}`",
        f"- retry-block KL learner-state `SARE` mean: `{retry_metrics['kl_lss_sare']['mean']:.4f}`",
        f"- frozen-comparable combined KL learner-state `SARE` mean: `{combined_metrics['kl_lss_sare']['mean']:.4f}`",
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
            "notes": f"hard-family successor candidate `{best_candidate}`",
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
            "hard_family_dev_sare_mean": dev_summary["candidate_mean"],
            "hard_family_dev_token_mean": dev_summary["token_mean"],
            "hard_family_holdout_sare_mean": holdout_summary["candidate_mean"],
            "hard_family_holdout_token_mean": holdout_summary["token_mean"],
            "route_validation_pass": bool(stage6_payload.get("stage6_pass")),
            "stability_pass": bool(stage7_payload.get("stage7_pass")),
            "healthy_block_sare_mean": _float(stage5_payload.get("candidate_healthy_sare_mean")),
        },
    }
    successor_json.parent.mkdir(parents=True, exist_ok=True)
    successor_json.write_text(json.dumps(candidate_pack, indent=2, sort_keys=True), encoding="utf-8")

    successor_lines = [
        "# Hard-Family Successor Candidate Pack",
        "",
        f"- candidate: `{best_candidate}`",
        "- status: `successor candidate pack only; not yet canonical until the decision memo says so`",
        f"- current thaw-qualified pack: `{campaign['current_candidate_pack']}`",
        f"- frozen benchmark pack: `{campaign['frozen_pack']}`",
        "",
        "## Stronger-Than-Current Summary",
        "",
        f"- hard-family dev KL learner-state `SARE` mean: `{dev_summary['candidate_mean']:.4f}` vs matched token_dense `{dev_summary['token_mean']:.4f}`",
        f"- hard-family holdout KL learner-state `SARE` mean: `{holdout_summary['candidate_mean']:.4f}` vs matched token_dense `{holdout_summary['token_mean']:.4f}`",
        f"- retry-block KL learner-state `SARE` mean: `{retry_metrics['kl_lss_sare']['mean']:.4f}` vs current thaw-qualified pack `{current_retry:.4f}`",
        f"- frozen-comparable combined KL learner-state `SARE` mean: `{combined_metrics['kl_lss_sare']['mean']:.4f}` vs current thaw-qualified pack `{current_combined:.4f}`",
        f"- route/stability status: `{'pass' if stage6_payload.get('stage6_pass') and stage7_payload.get('stage7_pass') else 'mixed'}`",
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


def _write_stop_stage8(reports: dict[str, Any], reason: str, stage: str) -> None:
    successor_md = Path(reports["successor_pack_markdown"])
    successor_json = Path(reports["successor_pack_json"])
    gate_md = Path(reports["gate_report_markdown"])
    gate_json = Path(reports["gate_report_json"])
    for path in (successor_md, successor_json, gate_md, gate_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    successor_md.write_text(f"# Hard-Family Successor Candidate Pack\n\n- status: `not generated`\n- reason: `{reason}`\n", encoding="utf-8")
    successor_json.write_text(json.dumps({"status": "not_generated", "reason": reason, "stage": stage}, indent=2, sort_keys=True), encoding="utf-8")
    gate_md.write_text(f"# Hard-Family Gate Report\n\n- status: `not run`\n- reason: `{reason}`\n", encoding="utf-8")
    gate_json.write_text(json.dumps({"status": "not_run", "reason": reason, "stage": stage}, indent=2, sort_keys=True), encoding="utf-8")


def _render_decision_memo(stage2_payload: dict[str, Any], stage3_payload: dict[str, Any] | None, stage4_payload: dict[str, Any] | None, stage5_payload: dict[str, Any] | None, stage6_payload: dict[str, Any] | None, stage7_payload: dict[str, Any] | None, gate_payload: dict[str, Any] | None, output: Path) -> None:
    best_candidate = None if stage3_payload is None else stage3_payload.get("best_candidate")
    if gate_payload is None:
        gate_verdict = None
    else:
        verdict = gate_payload.get("verdict")
        gate_verdict = None if verdict is None else str(verdict)
    stop_stage = None
    final_status = "remains thaw-qualified but not canonical"
    if not stage2_payload.get("advancing_candidates"):
        stop_stage = "Stage 2"
    elif not best_candidate:
        stop_stage = "Stage 3"
    elif stage4_payload is None or not stage4_payload.get("stage4_pass"):
        stop_stage = "Stage 4"
    elif stage5_payload is None or not stage5_payload.get("stage5_pass"):
        stop_stage = "Stage 5"
    elif stage6_payload is None or not stage6_payload.get("stage6_pass"):
        stop_stage = "Stage 6"
    elif stage7_payload is None or not stage7_payload.get("stage7_pass"):
        stop_stage = "Stage 7"
    elif gate_payload is None:
        stop_stage = "Stage 8"
    else:
        dev_summary = next(row for row in stage3_payload.get("candidate_summaries", []) if str(row["candidate"]) == str(best_candidate))
        holdout = stage4_payload.get("holdout_summary") or {}
        hard_family_fixed = _float(dev_summary.get("candidate_minus_token")) >= 0.0 and _float(holdout.get("candidate_minus_token")) >= 0.0
        single_ok = _float(dev_summary.get("candidate_minus_single")) >= -0.02 and _float(holdout.get("candidate_minus_single")) >= -0.02
        if gate_verdict == "PASS: thaw consideration allowed" and hard_family_fixed and single_ok:
            final_status = "canonical within DoorKey"
        elif gate_verdict != "PASS: thaw consideration allowed":
            final_status = "falls back below thaw-qualified"
            stop_stage = "Stage 8"
    lines = [
        "# Hard-Family Canonization Decision Memo",
        "",
        f"- stop stage: `{stop_stage or 'none'}`",
        f"- final status: `{final_status}`",
        f"- gate verdict: `{gate_verdict or 'not run'}`",
        "",
        "## Stage Summary",
        "",
        f"- Stage 2 advancing candidates: `{stage2_payload.get('advancing_candidates', [])}`",
        f"- Stage 3 best candidate: `{best_candidate}`",
        f"- Stage 4 holdout pass: `{None if stage4_payload is None else stage4_payload.get('stage4_pass')}`",
        f"- Stage 5 anti-regression pass: `{None if stage5_payload is None else stage5_payload.get('stage5_pass')}`",
        f"- Stage 6 route pass: `{None if stage6_payload is None else stage6_payload.get('stage6_pass')}`",
        f"- Stage 7 stability pass: `{None if stage7_payload is None else stage7_payload.get('stage7_pass')}`",
        "",
        "## Final Result",
        "",
    ]
    if final_status == "canonical within DoorKey":
        lines.append("- The long hard-family program fixes the hard-family blocker on both development and holdout splits, preserves the broader DoorKey picture, stays routed and stable, and clears the pack-based gate strongly enough to recommend canonization within DoorKey only.")
    elif final_status == "falls back below thaw-qualified":
        lines.append("- The expanded hard-family program produced a successor candidate pack that no longer clears the frozen benchmark gate strongly enough to keep thaw-qualified status.")
    else:
        lines.append("- The long hard-family program did not remove the remaining hard-family blocker strongly enough to replace the current thaw-qualified candidate. The accepted state remains thaw-qualified but not canonical.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Long hard-family qualification program for DoorKey.")
    sub = parser.add_subparsers(dest="command", required=True)

    definition = sub.add_parser("definition")
    definition.add_argument("--campaign-config", required=True)
    definition.add_argument("--output", required=True)

    registration = sub.add_parser("registration")
    registration.add_argument("--campaign-config", required=True)
    registration.add_argument("--output", required=True)

    baseline = sub.add_parser("baseline-sync")
    baseline.add_argument("--campaign-config", required=True)
    baseline.add_argument("--output", required=True)

    shortlist = sub.add_parser("shortlist")
    shortlist.add_argument("--campaign-config", required=True)
    shortlist.add_argument("--output", required=True)

    stage2 = sub.add_parser("stage2-report")
    stage2.add_argument("--campaign-config", required=True)
    stage2.add_argument("--output", required=True)
    stage2.add_argument("--csv", required=True)
    stage2.add_argument("--json", required=True)

    stage3 = sub.add_parser("stage3-report")
    stage3.add_argument("--campaign-config", required=True)
    stage3.add_argument("--stage2-json", required=True)
    stage3.add_argument("--output", required=True)
    stage3.add_argument("--csv", required=True)
    stage3.add_argument("--json", required=True)

    stage4 = sub.add_parser("stage4-report")
    stage4.add_argument("--campaign-config", required=True)
    stage4.add_argument("--stage3-json", required=True)
    stage4.add_argument("--output", required=True)
    stage4.add_argument("--csv", required=True)
    stage4.add_argument("--json", required=True)

    stage5 = sub.add_parser("stage5-report")
    stage5.add_argument("--campaign-config", required=True)
    stage5.add_argument("--stage3-json", required=True)
    stage5.add_argument("--stage4-json", required=True)
    stage5.add_argument("--output", required=True)
    stage5.add_argument("--csv", required=True)
    stage5.add_argument("--json", required=True)

    stage6 = sub.add_parser("stage6-report")
    stage6.add_argument("--campaign-config", required=True)
    stage6.add_argument("--stage3-json", required=True)
    stage6.add_argument("--stage4-json", required=True)
    stage6.add_argument("--stage5-json", required=True)
    stage6.add_argument("--route-csv", required=True)
    stage6.add_argument("--output", required=True)
    stage6.add_argument("--csv", required=True)
    stage6.add_argument("--json", required=True)

    stage7 = sub.add_parser("stage7-report")
    stage7.add_argument("--campaign-config", required=True)
    stage7.add_argument("--stage3-json", required=True)
    stage7.add_argument("--stage4-json", required=True)
    stage7.add_argument("--stage5-json", required=True)
    stage7.add_argument("--stage6-json", required=True)
    stage7.add_argument("--device", required=True)
    stage7.add_argument("--episodes", type=int, required=True)
    stage7.add_argument("--output", required=True)
    stage7.add_argument("--csv", required=True)
    stage7.add_argument("--json", required=True)

    stage8 = sub.add_parser("successor-pack")
    stage8.add_argument("--campaign-config", required=True)
    stage8.add_argument("--stage3-json", required=True)
    stage8.add_argument("--stage4-json", required=True)
    stage8.add_argument("--stage5-json", required=True)
    stage8.add_argument("--stage6-json", required=True)
    stage8.add_argument("--stage7-json", required=True)
    stage8.add_argument("--summary-output", required=True)
    stage8.add_argument("--metrics-output", required=True)
    stage8.add_argument("--combined-report-output", required=True)
    stage8.add_argument("--combined-report-csv", required=True)
    stage8.add_argument("--retry-report-output", required=True)
    stage8.add_argument("--retry-report-csv", required=True)
    stage8.add_argument("--successor-pack-markdown", required=True)
    stage8.add_argument("--successor-pack-json", required=True)

    decision = sub.add_parser("decision-memo")
    decision.add_argument("--stage2-json", required=True)
    decision.add_argument("--stage3-json", required=True)
    decision.add_argument("--stage4-json", required=True)
    decision.add_argument("--stage5-json", required=True)
    decision.add_argument("--stage6-json", required=True)
    decision.add_argument("--stage7-json", required=True)
    decision.add_argument("--gate-json", default=None)
    decision.add_argument("--output", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command in {"definition", "registration", "baseline-sync", "shortlist", "stage2-report"}:
        campaign = _load_yaml(Path(args.campaign_config))
        if args.command == "definition":
            _render_definition(campaign, Path(args.output))
            return
        if args.command == "registration":
            _render_registration(campaign, Path(args.output))
            return
        if args.command == "baseline-sync":
            _render_baseline_sync(campaign, Path(args.output))
            return
        if args.command == "shortlist":
            _render_shortlist(campaign, Path(args.output))
            return
        if args.command == "stage2-report":
            _render_stage2(campaign, Path(args.output), Path(args.csv), Path(args.json))
            return
    if args.command == "stage3-report":
        campaign = _load_yaml(Path(args.campaign_config))
        _render_stage3(campaign, _read_json(Path(args.stage2_json)), Path(args.output), Path(args.csv), Path(args.json))
        return
    if args.command == "stage4-report":
        campaign = _load_yaml(Path(args.campaign_config))
        _render_stage4(campaign, _read_json(Path(args.stage3_json)), Path(args.output), Path(args.csv), Path(args.json))
        return
    if args.command == "stage5-report":
        campaign = _load_yaml(Path(args.campaign_config))
        _render_stage5(campaign, _read_json(Path(args.stage3_json)), _read_json(Path(args.stage4_json)), Path(args.output), Path(args.csv), Path(args.json))
        return
    if args.command == "stage6-report":
        campaign = _load_yaml(Path(args.campaign_config))
        _render_stage6(
            campaign,
            _read_json(Path(args.stage3_json)),
            _read_json(Path(args.stage4_json)),
            _read_json(Path(args.stage5_json)),
            _read_csv_rows(Path(args.route_csv)),
            Path(args.output),
            Path(args.csv),
            Path(args.json),
        )
        return
    if args.command == "stage7-report":
        campaign = _load_yaml(Path(args.campaign_config))
        _render_stage7(
            campaign,
            _read_json(Path(args.stage3_json)),
            _read_json(Path(args.stage4_json)),
            _read_json(Path(args.stage5_json)),
            _read_json(Path(args.stage6_json)),
            args.device,
            args.episodes,
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
        stage6_payload = _read_json(Path(args.stage6_json))
        stage7_payload = _read_json(Path(args.stage7_json))
        best_candidate = stage3_payload.get("best_candidate")
        if not best_candidate:
            _write_stop_stage8(campaign["reports"], "no candidate survived dev fairness", "Stage 3")
            return
        if not stage4_payload.get("stage4_pass"):
            _write_stop_stage8(campaign["reports"], "candidate did not survive hard-family holdout evaluation", "Stage 4")
            return
        if not stage5_payload.get("stage5_pass"):
            _write_stop_stage8(campaign["reports"], "candidate did not survive healthy-block anti-regression", "Stage 5")
            return
        if not stage6_payload.get("stage6_pass"):
            _write_stop_stage8(campaign["reports"], "candidate did not survive route validation", "Stage 6")
            return
        if not stage7_payload.get("stage7_pass"):
            _write_stop_stage8(campaign["reports"], "candidate did not survive stability validation", "Stage 7")
            return
        _render_successor_pack(
            campaign,
            stage3_payload,
            stage4_payload,
            stage5_payload,
            stage6_payload,
            stage7_payload,
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
        gate_payload = None if args.gate_json is None or not Path(args.gate_json).exists() else _read_json(Path(args.gate_json))
        _render_decision_memo(
            _read_json(Path(args.stage2_json)),
            _read_json(Path(args.stage3_json)),
            _read_json(Path(args.stage4_json)),
            _read_json(Path(args.stage5_json)),
            _read_json(Path(args.stage6_json)),
            _read_json(Path(args.stage7_json)),
            gate_payload,
            Path(args.output),
        )
        return
    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
