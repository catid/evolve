from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from psmn_rl.analysis.benchmark_pack import sha256_path
from psmn_rl.analysis.claim_gate import evaluate_pack_claim_gate_from_paths, render_pack_gate_report
from psmn_rl.analysis.lss_frozen_claim import REPRO_MODES, _evaluate_modes
from psmn_rl.analysis.lss_robustness import EvalTarget, _evaluate_targets, _format_float
from psmn_rl.utils.io import get_git_commit, get_git_dirty


LABELS: tuple[str, ...] = (
    "recovered_token_dense",
    "kl_lss_token_dense",
    "kl_lss_single_expert",
    "baseline_sare",
    "kl_lss_sare",
)
DISPLAY_NAMES: dict[str, str] = {
    "recovered_token_dense": "recovered token_dense",
    "kl_lss_token_dense": "KL learner-state token_dense",
    "kl_lss_single_expert": "KL learner-state single_expert",
    "baseline_sare": "baseline PPO SARE",
    "kl_lss_sare": "KL learner-state SARE",
}
NEW_BLOCK_LABELS: tuple[str, ...] = ("post_pass_a", "post_pass_b")


@dataclass(slots=True)
class CandidateRun:
    lane: str
    seed: int
    label: str
    run_dir: Path
    summary: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "-"):
        return default
    return float(value)


def _int(value: Any, default: int = 0) -> int:
    if value in (None, "", "-"):
        return default
    return int(value)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "complete_seed_failures": 0.0, "seed_count": 0.0}
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "complete_seed_failures": float(sum(1 for value in values if value <= 0.0)),
        "seed_count": float(len(values)),
    }


def _artifact(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def _lane_seed_list(group: dict[str, Any]) -> list[tuple[str, int]]:
    if "lane" in group:
        return [(str(group["lane"]), int(seed)) for seed in group.get("seeds", [])]
    result: list[tuple[str, int]] = []
    for block in group.get("blocks", []):
        lane = str(block["lane"])
        result.extend((lane, int(seed)) for seed in block.get("seeds", []))
    return result


def _summary_round(summary: dict[str, Any], round_index: int | None) -> dict[str, Any]:
    if round_index is None or round_index <= 0:
        return {}
    rounds = summary.get("rounds", [])
    if round_index > len(rounds):
        return {}
    return dict(rounds[round_index - 1])


def _discover_new_block_targets(campaign: dict[str, Any], root: Path) -> list[EvalTarget]:
    targets: list[EvalTarget] = []
    for group_name in ("new_block_a", "new_block_b"):
        group = campaign["seed_groups"][group_name]
        lane = str(group["lane"])
        for seed in group["seeds"]:
            seed_root = root / lane / f"seed_{seed}"
            baseline_specs = [
                ("recovered_token_dense", "token_dense_ent1e3", "token_dense", "baseline_recovered_token_dense"),
                ("baseline_sare", "sare_ent1e3", "sare", "baseline_sare"),
            ]
            for label, run_name, variant, method in baseline_specs:
                run_dir = seed_root / run_name
                targets.append(
                    EvalTarget(
                        seed=int(seed),
                        label=label,
                        variant=variant,
                        config_path=run_dir / "resolved_config.yaml",
                        checkpoint_path=run_dir / "latest.pt",
                        run_dir=run_dir,
                        method=method,
                        stage=lane,
                        metadata={"lane": lane, "block": lane},
                        command_path=run_dir / "command.txt",
                    )
                )
            lss_specs = [
                ("kl_lss_token_dense", "kl_lss_token_dense", "token_dense"),
                ("kl_lss_single_expert", "kl_lss_single_expert", "single_expert"),
                ("kl_lss_sare", "kl_lss_sare", "sare"),
            ]
            for label, run_name, variant in lss_specs:
                run_dir = seed_root / run_name
                targets.append(
                    EvalTarget(
                        seed=int(seed),
                        label=label,
                        variant=variant,
                        config_path=run_dir / "student_resolved_config.yaml",
                        checkpoint_path=run_dir / "latest.pt",
                        run_dir=run_dir,
                        method="post_unlock_weighted",
                        stage=lane,
                        metadata={"lane": lane, "block": lane},
                        command_path=run_dir / "command.txt",
                    )
                )
    return targets


def _group_eval_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["lane"]), int(row["seed"]), str(row["label"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def _summary_row_from_eval(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    greedy = next(row for row in group_rows if str(row["mode"]) == "greedy")
    sampled = next(row for row in group_rows if str(row["mode"]) == "sampled_t1.0")
    run_dir = Path(str(greedy["run_dir"]))
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    best_round_index = _int(summary.get("best_round_index")) if summary else 0
    best_round = _summary_round(summary, best_round_index)
    final_round = _summary_round(summary, len(summary.get("rounds", [])))
    return {
        "lane": str(greedy["lane"]),
        "block": str(greedy.get("block", greedy["lane"])),
        "seed": int(greedy["seed"]),
        "label": str(greedy["label"]),
        "variant": str(greedy["variant"]),
        "method": str(greedy["method"]),
        "config_path": str(greedy["config_path"]),
        "checkpoint_path": str(greedy["checkpoint_path"]),
        "run_dir": str(greedy["run_dir"]),
        "final_greedy_success": _float(greedy.get("eval_success_rate")),
        "final_sampled_success": _float(sampled.get("eval_success_rate")),
        "eval_return": _float(greedy.get("eval_return")),
        "route_entropy": _float(greedy.get("route_entropy")),
        "path_entropy": _float(greedy.get("path_entropy")),
        "active_compute_proxy": _float(greedy.get("active_compute_proxy")),
        "best_round_index": best_round_index,
        "best_round_greedy_success": _float(summary.get("best_round_greedy_success")),
        "best_round_disagreement": _float(best_round.get("collection/disagreement_rate")),
        "best_round_post_unlock_frac": _float(best_round.get("collection/phase_frac_post_unlock")),
        "best_round_route_entropy": _float(best_round.get("collection/route_entropy")),
        "best_round_path_entropy": _float(best_round.get("collection/path_entropy")),
        "final_round_disagreement": _float(final_round.get("collection/disagreement_rate")),
        "final_round_post_unlock_frac": _float(final_round.get("collection/phase_frac_post_unlock")),
        "final_round_route_entropy": _float(final_round.get("collection/route_entropy")),
        "final_round_path_entropy": _float(final_round.get("collection/path_entropy")),
        "final_round_active_compute_proxy": _float(final_round.get("collection/active_compute_proxy")),
    }


def _new_block_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for lane in NEW_BLOCK_LABELS:
        lane_rows = [row for row in summary_rows if str(row["lane"]) == lane]
        lane_stats = {
            label: _stats([_float(row["final_greedy_success"]) for row in lane_rows if str(row["label"]) == label])
            for label in LABELS
        }
        candidate_delta_single = lane_stats["kl_lss_sare"]["mean"] - lane_stats["kl_lss_single_expert"]["mean"]
        candidate_delta_token = lane_stats["kl_lss_sare"]["mean"] - lane_stats["kl_lss_token_dense"]["mean"]
        summaries.append(
            {
                "lane": lane,
                "candidate_mean": lane_stats["kl_lss_sare"]["mean"],
                "token_mean": lane_stats["kl_lss_token_dense"]["mean"],
                "single_mean": lane_stats["kl_lss_single_expert"]["mean"],
                "candidate_complete_seed_failures": lane_stats["kl_lss_sare"]["complete_seed_failures"],
                "candidate_delta_single_expert": candidate_delta_single,
                "candidate_delta_token_dense": candidate_delta_token,
                "competitive_vs_single_expert": candidate_delta_single >= -0.05,
            }
        )
    return summaries


def _render_registration(campaign: dict[str, Any], output: Path) -> None:
    lines = [
        "# Post-PASS Campaign Registration",
        "",
        f"- frozen manifest: `{campaign['frozen_manifest']}`",
        f"- frozen benchmark pack: `{campaign['frozen_pack']}`",
        f"- current thaw-qualified candidate pack: `{campaign['current_candidate_pack']}`",
        f"- current candidate: `{campaign['current_candidate_name']}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## New Seed Blocks",
        "",
        f"- `{campaign['seed_groups']['new_block_a']['lane']}`: `{campaign['seed_groups']['new_block_a']['seeds']}`",
        f"- `{campaign['seed_groups']['new_block_b']['lane']}`: `{campaign['seed_groups']['new_block_b']['seeds']}`",
        "",
        "## Fairness And Mechanism Plan",
        "",
        "- Stage 1: run the thaw-qualified `post_unlock_weighted` candidate plus matched post_unlock_weighted structured controls on both fresh blocks.",
        "- Stage 2: aggregate fairness across the historical campaign blocks and both new fresh blocks.",
        "- Stage 3: extend route validation to two additional historically strong recovered seeds and two historically weak retry-block seeds.",
        "- Stage 4: evaluate multiple candidate checkpoints on one weak representative and one stronger representative, with frozen-baseline stability side-by-side.",
        "- Stage 5: draft a successor benchmark pack without replacing the frozen pack.",
        "- Stage 6: package the expanded candidate, run the frozen-pack gate, and decide canonization vs thaw-only vs fallback.",
        "",
        "## Canonization Bar",
        "",
        "- The candidate must remain ahead of matched structured controls on both new fresh blocks.",
        "- The expanded DoorKey picture must preserve or improve the current combined candidate mean while staying inside the DoorKey-only teacher-guided envelope.",
        "- Route disruption must remain materially harmful on both added strong seeds and historically weak seeds.",
        "- The selected candidate checkpoints must not reduce to narrow one-checkpoint spikes.",
        "- The expanded candidate pack must still clear the existing frozen-pack gate.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_baseline_sync(campaign: dict[str, Any], output: Path) -> None:
    manifest = _load_yaml(Path(campaign["frozen_manifest"]))
    frozen_pack = _read_json(Path(campaign["frozen_pack"]))
    current_candidate_pack = _read_json(Path(campaign["current_candidate_pack"]))
    current_retry = current_candidate_pack["metrics"]["retry_block"]["kl_lss_sare"]["mean"]
    current_single = current_candidate_pack["metrics"]["retry_block"]["kl_lss_single_expert"]["mean"]
    current_combined = current_candidate_pack["metrics"]["combined"]["kl_lss_sare"]["mean"]
    lines = [
        "# Post-PASS Baseline Sync",
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
        "## Current Candidate Snapshot",
        "",
        f"- current retry-block KL learner-state `SARE` mean: `{current_retry:.4f}`",
        f"- current retry-block KL learner-state `single_expert` mean: `{current_single:.4f}`",
        f"- current combined KL learner-state `SARE` mean: `{current_combined:.4f}`",
        "",
        "## Interpretation",
        "",
        "- The frozen benchmark pack remains the comparison unit for this phase.",
        "- The `post_unlock_weighted` candidate is thaw-qualified relative to the frozen pack, but not yet canonicalized as a successor benchmark.",
        f"- frozen pack schema version: `{frozen_pack.get('schema_version')}`",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_stage1(summary_rows: list[dict[str, Any]], output: Path, csv_output: Path, json_output: Path) -> None:
    block_summaries = _new_block_summary(summary_rows)
    candidate_failures = sum(int(row["candidate_complete_seed_failures"]) for row in block_summaries)
    stage1_pass = bool(block_summaries) and all(bool(row["competitive_vs_single_expert"]) for row in block_summaries) and candidate_failures <= 1
    selected_strong_cases: list[dict[str, Any]] = []
    for lane in NEW_BLOCK_LABELS:
        candidate_rows = [row for row in summary_rows if str(row["lane"]) == lane and str(row["label"]) == "kl_lss_sare"]
        if not candidate_rows:
            continue
        selected = max(candidate_rows, key=lambda item: (_float(item["final_greedy_success"]), -int(item["seed"])))
        selected_strong_cases.append({"lane": lane, "seed": int(selected["seed"]), "run_dir": str(selected["run_dir"])})

    lines = [
        "# Post-PASS Stage 1 Fresh Blocks",
        "",
        "| Block | Seed | Variant | Greedy Success | Sampled t=1.0 Success | Route Entropy | Path Entropy | Active Compute |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary_rows, key=lambda item: (str(item["lane"]), int(item["seed"]), str(item["label"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["lane"]),
                    str(row["seed"]),
                    DISPLAY_NAMES[str(row["label"])],
                    _format_float(row["final_greedy_success"]),
                    _format_float(row["final_sampled_success"]),
                    _format_float(row["route_entropy"]),
                    _format_float(row["path_entropy"]),
                    _format_float(row["active_compute_proxy"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Block Summary",
            "",
            "| Block | candidate SARE Mean | matched token_dense Mean | matched single_expert Mean | candidate - single_expert | candidate - token_dense | Complete-Seed Failures | Stage 1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
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
                    _format_float(row["candidate_delta_single_expert"]),
                    _format_float(row["candidate_delta_token_dense"]),
                    str(int(row["candidate_complete_seed_failures"])),
                    "`pass`" if row["competitive_vs_single_expert"] else "`mixed`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- selected strong fresh-block cases for later route validation: `{[(row['lane'], row['seed']) for row in selected_strong_cases]}`",
            f"- stage-1 status: `{'pass' if stage1_pass else 'mixed'}`",
            "- The new fresh blocks use matched post_unlock_weighted learner-state runs for `token_dense`, `single_expert`, and `SARE`, plus recovered token_dense and baseline PPO SARE for context.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(csv_output, summary_rows)
    _write_json(
        json_output,
        {
            "stage": "stage1",
            "summary_rows": summary_rows,
            "block_summaries": block_summaries,
            "selected_strong_cases": selected_strong_cases,
            "stage1_pass": stage1_pass,
        },
    )


def _historical_candidate_rows(stage4_csv: Path, candidate_name: str) -> list[dict[str, Any]]:
    rows = _read_csv_rows(stage4_csv)
    return [row for row in rows if str(row.get("candidate")) == candidate_name and str(row.get("label")) in {"kl_lss_token_dense", "kl_lss_single_expert", "kl_lss_sare"}]


def _render_stage2(
    candidate_name: str,
    historical_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    output: Path,
    csv_output: Path,
    json_output: Path,
) -> None:
    detail_rows: list[dict[str, Any]] = []
    for row in historical_rows:
        detail_rows.append(
            {
                "source": "historical_candidate",
                "lane": str(row["lane"]),
                "block": str(row["lane"]),
                "seed": int(row["seed"]),
                "label": str(row["label"]),
                "final_greedy_success": _float(row["final_greedy_success"]),
                "run_dir": str(row["run_dir"]),
            }
        )
    for row in new_rows:
        if str(row["label"]) not in {"kl_lss_token_dense", "kl_lss_single_expert", "kl_lss_sare"}:
            continue
        detail_rows.append(
            {
                "source": "new_fresh_blocks",
                "lane": str(row["lane"]),
                "block": str(row["block"]),
                "seed": int(row["seed"]),
                "label": str(row["label"]),
                "final_greedy_success": _float(row["final_greedy_success"]),
                "run_dir": str(row["run_dir"]),
            }
        )

    block_summaries: list[dict[str, Any]] = []
    blocks = sorted({str(row["block"]) for row in detail_rows})
    for block in blocks:
        block_rows = [row for row in detail_rows if str(row["block"]) == block]
        stats = {
            label: _stats([_float(row["final_greedy_success"]) for row in block_rows if str(row["label"]) == label])
            for label in ("kl_lss_token_dense", "kl_lss_single_expert", "kl_lss_sare")
        }
        block_summaries.append(
            {
                "block": block,
                "candidate_mean": stats["kl_lss_sare"]["mean"],
                "token_mean": stats["kl_lss_token_dense"]["mean"],
                "single_mean": stats["kl_lss_single_expert"]["mean"],
                "candidate_minus_token": stats["kl_lss_sare"]["mean"] - stats["kl_lss_token_dense"]["mean"],
                "candidate_minus_single": stats["kl_lss_sare"]["mean"] - stats["kl_lss_single_expert"]["mean"],
                "candidate_failures": stats["kl_lss_sare"]["complete_seed_failures"],
                "single_failures": stats["kl_lss_single_expert"]["complete_seed_failures"],
            }
        )

    overall_stats = {
        label: _stats([_float(row["final_greedy_success"]) for row in detail_rows if str(row["label"]) == label])
        for label in ("kl_lss_token_dense", "kl_lss_single_expert", "kl_lss_sare")
    }
    stage2_competitive = (
        overall_stats["kl_lss_sare"]["mean"] >= overall_stats["kl_lss_single_expert"]["mean"] - 0.02
        and sum(1 for row in block_summaries if row["candidate_minus_single"] < -0.10) <= 1
    )

    lines = [
        "# Post-PASS Stage 2 Full Fairness",
        "",
        f"- candidate: `{candidate_name}`",
        "",
        "| Block | Seed | Variant | Greedy Success |",
        "| --- | --- | --- | ---: |",
    ]
    for row in sorted(detail_rows, key=lambda item: (str(item["block"]), int(item["seed"]), str(item["label"]))):
        lines.append(
            f"| {row['block']} | {row['seed']} | {DISPLAY_NAMES[str(row['label'])]} | {_format_float(row['final_greedy_success'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-Block Means",
            "",
            "| Block | candidate SARE | matched token_dense | matched single_expert | candidate - token_dense | candidate - single_expert | candidate complete-seed failures |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in block_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["block"]),
                    _format_float(row["candidate_mean"]),
                    _format_float(row["token_mean"]),
                    _format_float(row["single_mean"]),
                    _format_float(row["candidate_minus_token"]),
                    _format_float(row["candidate_minus_single"]),
                    str(int(row["candidate_failures"])),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Combined Means",
            "",
            f"- candidate KL learner-state `SARE`: `{overall_stats['kl_lss_sare']['mean']:.4f}`",
            f"- matched KL learner-state `token_dense`: `{overall_stats['kl_lss_token_dense']['mean']:.4f}`",
            f"- matched KL learner-state `single_expert`: `{overall_stats['kl_lss_single_expert']['mean']:.4f}`",
            "",
            "## Interpretation",
            "",
            f"- stage-2 competitiveness status: `{'pass' if stage2_competitive else 'mixed'}`",
            "- The expanded fairness view keeps the per-block structure visible so the qualification decision does not collapse into a single summary mean.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(csv_output, detail_rows)
    _write_json(
        json_output,
        {
            "stage": "stage2",
            "candidate": candidate_name,
            "detail_rows": detail_rows,
            "block_summaries": block_summaries,
            "overall_stats": overall_stats,
            "stage2_competitive": stage2_competitive,
        },
    )


def _candidate_run_dir(campaign: dict[str, Any], lane: str, seed: int) -> Path:
    if lane in NEW_BLOCK_LABELS:
        return Path(campaign["new_block_output_root"]) / lane / f"seed_{seed}" / "kl_lss_sare"
    if lane == "fresh_final":
        return Path(campaign["historical_candidate_roots"]["weak"]) / lane / f"seed_{seed}" / "kl_lss_sare"
    return Path(campaign["historical_candidate_roots"]["strong"]) / lane / f"seed_{seed}" / "kl_lss_sare"


def _baseline_sare_run_dir(campaign: dict[str, Any], lane: str, seed: int) -> Path:
    lane_cfg = campaign["baseline_lane_roots"][lane]
    return Path(lane_cfg["sare_lss_root"]) / f"seed_{seed}" / str(lane_cfg["sare_lss_run_name"])


def _summary_compare_row(campaign: dict[str, Any], lane: str, seed: int) -> dict[str, Any]:
    candidate_summary = _read_json(_candidate_run_dir(campaign, lane, seed) / "summary.json")
    baseline_summary = _read_json(_baseline_sare_run_dir(campaign, lane, seed) / "summary.json")
    candidate_best = _summary_round(candidate_summary, _int(candidate_summary.get("best_round_index")))
    baseline_best = _summary_round(baseline_summary, _int(baseline_summary.get("best_round_index")))
    candidate_final = _summary_round(candidate_summary, len(candidate_summary.get("rounds", [])))
    baseline_final = _summary_round(baseline_summary, len(baseline_summary.get("rounds", [])))
    return {
        "lane": lane,
        "seed": seed,
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
    }


def _render_stage3(campaign: dict[str, Any], route_rows: list[dict[str, Any]], output: Path, csv_output: Path, json_output: Path) -> None:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in route_rows:
        grouped.setdefault((str(row["lane"]), int(row["seed"])), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (lane, seed), case_rows in sorted(grouped.items()):
        baseline = next(row for row in case_rows if row["probe"] == "baseline")
        fixed = next(row for row in case_rows if row["probe"] == "router_override")
        random = next(row for row in case_rows if row["probe"] == "route_randomization")
        ablations = [row for row in case_rows if row["probe"] == "expert_ablation"]
        worst = min(ablations, key=lambda item: _float(item["eval_success_rate"])) if ablations else baseline
        summary_compare = _summary_compare_row(campaign, lane, seed)
        summaries.append(
            {
                "lane": lane,
                "seed": seed,
                "baseline_success": _float(baseline["eval_success_rate"]),
                "fixed_drop": _float(baseline["eval_success_rate"]) - _float(fixed["eval_success_rate"]),
                "random_drop": _float(baseline["eval_success_rate"]) - _float(random["eval_success_rate"]),
                "worst_ablation_drop": _float(baseline["eval_success_rate"]) - _float(worst["eval_success_rate"]),
                **summary_compare,
            }
        )
    fixed_ok = all(row["fixed_drop"] >= 0.25 for row in summaries)
    ablation_ok = all(row["worst_ablation_drop"] >= 0.25 for row in summaries)
    random_ok = sum(1 for row in summaries if row["random_drop"] >= 0.25) >= max(2, len(summaries) // 2)
    stage3_pass = bool(summaries) and fixed_ok and ablation_ok and random_ok

    lines = [
        "# Post-PASS Stage 3 Route Validation",
        "",
        "| Lane | Seed | Baseline | Fixed-Router Drop | Route-Randomization Drop | Worst Ablation Drop | Final Disagreement Δ | Final Post-Unlock Frac Δ | Best-Round Shift |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["lane"]),
                    str(row["seed"]),
                    _format_float(row["baseline_success"]),
                    _format_float(row["fixed_drop"]),
                    _format_float(row["random_drop"]),
                    _format_float(row["worst_ablation_drop"]),
                    _format_float(row["candidate_final_disagreement"] - row["baseline_final_disagreement"]),
                    _format_float(row["candidate_final_post_unlock_frac"] - row["baseline_final_post_unlock_frac"]),
                    _format_float(row["candidate_best_round_index"] - row["baseline_best_round_index"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- stage-3 route status: `{'pass' if stage3_pass else 'mixed'}`",
            "- The added strong cases use two additional historically strong recovered routed seeds, while the weak cases remain the historically mixed retry-block seeds `47` and `59`.",
            "- The summary deltas show whether post_unlock_weighted shifts late-phase disagreement, route entropy, and cleanup-round timing relative to the frozen learner-state SARE baseline on the same seed.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(csv_output, summaries)
    _write_json(
        json_output,
        {
            "stage": "stage3",
            "summaries": summaries,
            "stage3_pass": stage3_pass,
            "fixed_ok": fixed_ok,
            "ablation_ok": ablation_ok,
            "random_ok": random_ok,
        },
    )


def _build_stability_targets(campaign: dict[str, Any], stage1_payload: dict[str, Any]) -> tuple[list[EvalTarget], list[dict[str, Any]]]:
    historical_rows = _historical_candidate_rows(Path(campaign["historical_candidate_csv"]), campaign["current_candidate_name"])
    weak_candidates = [row for row in historical_rows if str(row["lane"]) == "fresh_final" and str(row["label"]) == "kl_lss_sare"]
    strong_candidates = [
        row
        for row in historical_rows
        if str(row["lane"]) in {"original", "fresh", "fresh_extra"} and str(row["label"]) == "kl_lss_sare"
    ]
    weak_case = max(weak_candidates, key=lambda item: _float(item["final_greedy_success"]))
    strong_case = max(strong_candidates, key=lambda item: _float(item["final_greedy_success"]))

    cases = [
        {"kind": "weak", "lane": "fresh_final", "seed": int(weak_case["seed"])},
        {"kind": "strong", "lane": str(strong_case["lane"]), "seed": int(strong_case["seed"])},
    ]
    targets: list[EvalTarget] = []
    for case in cases:
        lane = str(case["lane"])
        seed = int(case["seed"])
        for source in ("candidate", "baseline"):
            if source == "candidate":
                run_dir = _candidate_run_dir(campaign, lane, seed)
            else:
                if lane in NEW_BLOCK_LABELS:
                    continue
                run_dir = _baseline_sare_run_dir(campaign, lane, seed)
            summary = _read_json(run_dir / "summary.json")
            config_path = run_dir / "student_resolved_config.yaml"
            for round_index in range(1, len(summary.get("rounds", [])) + 1):
                targets.append(
                    EvalTarget(
                        seed=seed,
                        label=f"{source}_{lane}_{seed}_round_{round_index}",
                        variant="sare",
                        config_path=config_path,
                        checkpoint_path=run_dir / f"round_{round_index:02d}.pt",
                        run_dir=run_dir,
                        method=f"{source}_round_eval",
                        stage="post_pass_stability",
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


def _render_stage4(rows: list[dict[str, Any]], cases: list[dict[str, Any]], output: Path, csv_output: Path, json_output: Path) -> None:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["source"]), str(row["lane"]), int(row["seed"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for case in cases:
        lane = str(case["lane"])
        seed = int(case["seed"])
        for source in ("candidate", "baseline"):
            key = (source, lane, seed)
            if key not in grouped:
                continue
            ordered = sorted(grouped[key], key=lambda item: int(item["round_index"]))
            successes = [_float(row["eval_success_rate"]) for row in ordered]
            summaries.append(
                {
                    "source": source,
                    "lane": lane,
                    "seed": seed,
                    "case_kind": str(case["kind"]),
                    "best_success": max(successes) if successes else 0.0,
                    "final_success": successes[-1] if successes else 0.0,
                    "round_successes": successes,
                    "classification": _stability_class(successes),
                }
            )
    candidate_rows = [row for row in summaries if row["source"] == "candidate"]
    stage4_pass = all(row["classification"] != "narrow_spike" for row in candidate_rows)

    lines = [
        "# Post-PASS Stage 4 Longitudinal Stability",
        "",
        "| Case | Lane | Seed | Source | Round Successes | Classification |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        success_str = ", ".join(f"{value:.4f}" for value in row["round_successes"])
        lines.append(
            f"| {row['case_kind']} | {row['lane']} | {row['seed']} | {row['source']} | `{success_str}` | `{row['classification']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- stage-4 stability status: `{'pass' if stage4_pass else 'mixed'}`",
            "- The weak representative tracks the historically difficult retry block, while the stronger representative comes from the healthiest historical routed seed retained by the candidate.",
            "- Candidate stability is treated as suspect if it reduces to a narrow one-checkpoint spike even when the gate metrics still clear.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(csv_output, rows)
    _write_json(
        json_output,
        {
            "stage": "stage4",
            "cases": cases,
            "summaries": summaries,
            "stage4_pass": stage4_pass,
        },
    )


def _render_successor_draft(
    campaign: dict[str, Any],
    stage2_payload: dict[str, Any],
    stage3_payload: dict[str, Any],
    stage4_payload: dict[str, Any],
    output: Path,
    json_output: Path,
) -> None:
    artifacts = [
        _artifact(Path(campaign["reports"]["stage2_report"]), "expanded_fairness_report"),
        _artifact(Path(campaign["reports"]["stage2_csv"]), "expanded_fairness_csv"),
        _artifact(Path(campaign["reports"]["stage3_report"]), "route_validation_report"),
        _artifact(Path(campaign["reports"]["stage3_csv"]), "route_validation_csv"),
        _artifact(Path(campaign["reports"]["stage4_report"]), "stability_report"),
        _artifact(Path(campaign["reports"]["stage4_csv"]), "stability_csv"),
    ]
    payload = {
        "schema_version": 1,
        "pack_type": "draft_successor_benchmark_pack",
        "status": "draft_not_canonical",
        "candidate_name": campaign["current_candidate_name"],
        "base_frozen_pack": {
            "path": str(campaign["frozen_pack"]),
            "sha256": sha256_path(Path(campaign["frozen_pack"])),
        },
        "provenance": {"git_commit": get_git_commit(), "git_dirty": get_git_dirty()},
        "expanded_fairness": stage2_payload.get("overall_stats", {}),
        "route_validation": {"stage3_pass": stage3_payload.get("stage3_pass"), "case_count": len(stage3_payload.get("summaries", []))},
        "stability": {"stage4_pass": stage4_payload.get("stage4_pass"), "summaries": stage4_payload.get("summaries", [])},
        "artifacts": artifacts,
        "notes": "Draft successor benchmark pack for post-pass qualification only. Not canonical in this phase.",
    }
    lines = [
        "# Post-PASS Successor Pack Draft",
        "",
        f"- candidate: `{campaign['current_candidate_name']}`",
        "- status: `draft successor pack only`",
        f"- base frozen pack: `{campaign['frozen_pack']}`",
        "",
        "## Expanded Candidate Summary",
        "",
        f"- expanded KL learner-state `SARE` mean: `{stage2_payload.get('overall_stats', {}).get('kl_lss_sare', {}).get('mean', 0.0):.4f}`",
        f"- expanded matched `single_expert` mean: `{stage2_payload.get('overall_stats', {}).get('kl_lss_single_expert', {}).get('mean', 0.0):.4f}`",
        f"- expanded matched `token_dense` mean: `{stage2_payload.get('overall_stats', {}).get('kl_lss_token_dense', {}).get('mean', 0.0):.4f}`",
        f"- route validation status: `{'pass' if stage3_payload.get('stage3_pass') else 'mixed'}`",
        f"- stability status: `{'pass' if stage4_payload.get('stage4_pass') else 'mixed'}`",
        "",
        "## Artifact Hashes",
        "",
        "| Role | Path | SHA256 |",
        "| --- | --- | --- |",
    ]
    for artifact in artifacts:
        lines.append(f"| {artifact['role']} | `{artifact['path']}` | `{artifact['sha256']}` |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(json_output, payload)


def _candidate_pack_rows(
    frozen_combined_csv: Path,
    historical_candidate_csv: Path,
    stage1_csv: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_rows = [
        row
        for row in _read_csv_rows(frozen_combined_csv)
        if str(row.get("mode")) == "greedy" and str(row.get("label")) in {"recovered_token_dense", "baseline_sare"}
    ]
    historical_rows = [
        row
        for row in _read_csv_rows(historical_candidate_csv)
        if str(row.get("candidate")) == "post_unlock_weighted" and str(row.get("label")) in {"kl_lss_token_dense", "kl_lss_single_expert", "kl_lss_sare"}
    ]
    new_rows = [
        row
        for row in _read_csv_rows(stage1_csv)
        if str(row.get("label")) in LABELS
    ]
    frozen_rows: list[dict[str, Any]] = []
    for row in baseline_rows:
        frozen_rows.append(
            {
                "lane": str(row["lane"]),
                "seed": int(row["seed"]),
                "label": str(row["label"]),
                "eval_success_rate": _float(row["eval_success_rate"]),
                "run_dir": str(row["run_dir"]),
            }
        )
    for row in historical_rows:
        frozen_rows.append(
            {
                "lane": str(row["lane"]),
                "seed": int(row["seed"]),
                "label": str(row["label"]),
                "eval_success_rate": _float(row["final_greedy_success"]),
                "run_dir": str(row["run_dir"]),
            }
        )
    expanded_rows = list(frozen_rows)
    for row in new_rows:
        expanded_rows.append(
            {
                "lane": str(row["lane"]),
                "seed": int(row["seed"]),
                "label": str(row["label"]),
                "eval_success_rate": _float(row["final_greedy_success"]),
                "run_dir": str(row["run_dir"]),
            }
        )
    retry_rows = [row for row in frozen_rows if str(row["lane"]) == "fresh_final"]
    return frozen_rows, retry_rows, expanded_rows


def _metrics_block(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {label: _stats([_float(row["eval_success_rate"]) for row in rows if str(row["label"]) == label]) for label in LABELS}


def _render_candidate_pack(
    campaign: dict[str, Any],
    combined_rows: list[dict[str, Any]],
    retry_rows: list[dict[str, Any]],
    expanded_rows: list[dict[str, Any]],
    summary_md: Path,
    metrics_json: Path,
    combined_md: Path,
    combined_csv: Path,
    retry_md: Path,
    retry_csv: Path,
    candidate_pack_output: Path,
) -> None:
    combined_metrics = _metrics_block(combined_rows)
    retry_metrics = _metrics_block(retry_rows)

    combined_lines = [
        "# Post-PASS Candidate Frozen-Comparable Combined DoorKey Report",
        "",
        f"- candidate: `{campaign['current_candidate_name']}`",
        "- scope: `frozen-comparable combined lane/seed set only`",
        f"- expanded qualification report: `{campaign['reports']['stage2_report']}`",
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
        "# Post-PASS Candidate Retry Block Report",
        "",
        "| Seed | recovered token_dense | KL learner-state token_dense | KL learner-state single_expert | baseline PPO SARE | KL learner-state SARE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lookup = {(str(row["lane"]), int(row["seed"]), str(row["label"])): row for row in retry_rows}
    for seed in campaign["seed_groups"]["retry_block"]["seeds"]:
        retry_lines.append(
            "| "
            + " | ".join(
                [
                    str(seed),
                    _format_float(lookup[("fresh_final", int(seed), "recovered_token_dense")]["eval_success_rate"]),
                    _format_float(lookup[("fresh_final", int(seed), "kl_lss_token_dense")]["eval_success_rate"]),
                    _format_float(lookup[("fresh_final", int(seed), "kl_lss_single_expert")]["eval_success_rate"]),
                    _format_float(lookup[("fresh_final", int(seed), "baseline_sare")]["eval_success_rate"]),
                    _format_float(lookup[("fresh_final", int(seed), "kl_lss_sare")]["eval_success_rate"]),
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
    _write_csv(retry_csv, retry_rows)

    summary_lines = [
        "# Post-PASS Candidate Summary",
        "",
        f"- candidate: `{campaign['current_candidate_name']}`",
        f"- retry-block KL learner-state `SARE` mean: `{retry_metrics['kl_lss_sare']['mean']:.4f}`",
        f"- retry-block KL learner-state `single_expert` mean: `{retry_metrics['kl_lss_single_expert']['mean']:.4f}`",
        f"- frozen-comparable combined KL learner-state `SARE` mean: `{combined_metrics['kl_lss_sare']['mean']:.4f}`",
        f"- expanded post-pass combined KL learner-state `SARE` mean: `{_stats([_float(row['eval_success_rate']) for row in expanded_rows if str(row['label']) == 'kl_lss_sare'])['mean']:.4f}`",
        f"- expanded qualification report: `{campaign['reports']['stage2_report']}`",
    ]
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    metrics_payload = {
        "schema_version": 1,
        "candidate_name": campaign["current_candidate_name"],
        "task": "DoorKey",
        "evaluation": campaign["evaluation"],
        "requested_claims": ["bounded_teacher_guided_doorkey_sare"],
        "controls_present": list(LABELS),
        "metrics": {"combined": combined_metrics, "retry_block": retry_metrics},
        "actual_sets": {
            "combined_lane_seeds": [[str(row["lane"]), int(row["seed"])] for row in combined_rows if str(row["label"]) == "kl_lss_sare"],
            "retry_block_lane_seeds": [[str(row["lane"]), int(row["seed"])] for row in retry_rows if str(row["label"]) == "kl_lss_sare"],
        },
        "provenance": {
            "git_commit": get_git_commit(),
            "git_dirty": get_git_dirty(),
            "notes": "post-pass qualification candidate `post_unlock_weighted`; gate-comparable metrics use the frozen combined set while expanded qualification evidence lives in the post-pass reports",
        },
    }
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    frozen_pack_path = Path(campaign["frozen_pack"])
    frozen_pack = _read_json(frozen_pack_path)
    candidate_pack = {
        "schema_version": 1,
        "pack_type": "candidate_result_pack",
        "candidate_name": campaign["current_candidate_name"],
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
        "actual_sets": metrics_payload["actual_sets"],
        "artifacts": [
            _artifact(summary_md, "candidate_summary_markdown"),
            _artifact(metrics_json, "candidate_metrics_json"),
            _artifact(combined_md, "combined_report_markdown"),
            _artifact(combined_csv, "combined_report_csv"),
            _artifact(retry_md, "retry_block_report_markdown"),
            _artifact(retry_csv, "retry_block_report_csv"),
        ],
        "provenance": metrics_payload["provenance"],
    }
    candidate_pack_output.parent.mkdir(parents=True, exist_ok=True)
    candidate_pack_output.write_text(json.dumps(candidate_pack, indent=2, sort_keys=True), encoding="utf-8")


def _render_decision_memo(
    stage1_payload: dict[str, Any],
    stage2_payload: dict[str, Any],
    stage3_payload: dict[str, Any],
    stage4_payload: dict[str, Any],
    gate_payload: dict[str, Any],
    output: Path,
) -> None:
    gate_verdict = str(gate_payload.get("verdict"))
    new_block_summaries = stage1_payload.get("block_summaries", [])
    stage1_canon_ok = all(
        _float(row["candidate_mean"]) > max(_float(row["single_mean"]), _float(row["token_mean"]))
        for row in new_block_summaries
    )
    stage2_canon_ok = (
        _float(stage2_payload.get("overall_stats", {}).get("kl_lss_sare", {}).get("mean")) >
        max(
            _float(stage2_payload.get("overall_stats", {}).get("kl_lss_single_expert", {}).get("mean")),
            _float(stage2_payload.get("overall_stats", {}).get("kl_lss_token_dense", {}).get("mean")),
        )
    )
    stage3_ok = bool(stage3_payload.get("stage3_pass"))
    stage4_ok = bool(stage4_payload.get("stage4_pass"))

    if gate_verdict != "PASS: thaw consideration allowed":
        final_status = "falls back to frozen-only candidate"
    elif stage1_canon_ok and stage2_canon_ok and stage3_ok and stage4_ok:
        final_status = "qualified for canonization within DoorKey"
    else:
        final_status = "remains thaw-qualified but not canonical"

    lines = [
        "# Post-PASS Canonization Decision Memo",
        "",
        f"- gate verdict: `{gate_verdict}`",
        f"- final status: `{final_status}`",
        "",
        "## Stage Summary",
        "",
        f"- Stage 1 fresh-block status: `{'pass' if stage1_payload.get('stage1_pass') else 'mixed'}`",
        f"- Stage 2 fairness status: `{'pass' if stage2_payload.get('stage2_competitive') else 'mixed'}`",
        f"- Stage 3 route status: `{'pass' if stage3_ok else 'mixed'}`",
        f"- Stage 4 stability status: `{'pass' if stage4_ok else 'mixed'}`",
        "",
        "## Canonization Bars",
        "",
        f"- ahead of both matched structured controls on both new blocks: `{stage1_canon_ok}`",
        f"- ahead of both structured controls on the expanded combined picture: `{stage2_canon_ok}`",
        f"- route dependence still meaningful on added strong/weak probes: `{stage3_ok}`",
        f"- representative checkpoints avoid narrow-spike instability: `{stage4_ok}`",
        "",
        "## Final Result",
        "",
    ]
    if final_status == "qualified for canonization within DoorKey":
        lines.append("- The candidate survives the post-pass qualification campaign strongly enough to recommend canonization within DoorKey only. The frozen pack remains the older comparison unit until a successor sealing step is completed.")
    elif final_status == "remains thaw-qualified but not canonical":
        lines.append("- The candidate still clears the frozen-pack gate, but the longer qualification campaign leaves enough fairness, strength-of-control, or stability ambiguity that it should remain thaw-qualified rather than canonical.")
    else:
        lines.append("- The expanded qualification campaign no longer supports advancing beyond the frozen benchmark baseline. The result should fall back to candidate-only status and not be treated as canonical.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-pass DoorKey qualification campaign.")
    sub = parser.add_subparsers(dest="command", required=True)

    registration = sub.add_parser("registration")
    registration.add_argument("--campaign-config", required=True)
    registration.add_argument("--output", required=True)

    baseline = sub.add_parser("baseline-sync")
    baseline.add_argument("--campaign-config", required=True)
    baseline.add_argument("--output", required=True)

    stage1 = sub.add_parser("stage1-report")
    stage1.add_argument("--campaign-config", required=True)
    stage1.add_argument("--stage1-root", required=True)
    stage1.add_argument("--device", required=True)
    stage1.add_argument("--episodes", type=int, required=True)
    stage1.add_argument("--output", required=True)
    stage1.add_argument("--csv", required=True)
    stage1.add_argument("--json", required=True)

    stage2 = sub.add_parser("stage2-report")
    stage2.add_argument("--campaign-config", required=True)
    stage2.add_argument("--stage1-csv", required=True)
    stage2.add_argument("--output", required=True)
    stage2.add_argument("--csv", required=True)
    stage2.add_argument("--json", required=True)

    stage3 = sub.add_parser("stage3-report")
    stage3.add_argument("--campaign-config", required=True)
    stage3.add_argument("--route-csv", required=True)
    stage3.add_argument("--output", required=True)
    stage3.add_argument("--csv", required=True)
    stage3.add_argument("--json", required=True)

    stage4 = sub.add_parser("stage4-report")
    stage4.add_argument("--campaign-config", required=True)
    stage4.add_argument("--stage1-json", required=True)
    stage4.add_argument("--device", required=True)
    stage4.add_argument("--episodes", type=int, required=True)
    stage4.add_argument("--output", required=True)
    stage4.add_argument("--csv", required=True)
    stage4.add_argument("--json", required=True)

    stage5 = sub.add_parser("successor-draft")
    stage5.add_argument("--campaign-config", required=True)
    stage5.add_argument("--stage2-json", required=True)
    stage5.add_argument("--stage3-json", required=True)
    stage5.add_argument("--stage4-json", required=True)
    stage5.add_argument("--output", required=True)
    stage5.add_argument("--json", required=True)

    pack = sub.add_parser("candidate-pack")
    pack.add_argument("--campaign-config", required=True)
    pack.add_argument("--stage1-csv", required=True)
    pack.add_argument("--summary-output", required=True)
    pack.add_argument("--metrics-output", required=True)
    pack.add_argument("--combined-report-output", required=True)
    pack.add_argument("--combined-report-csv", required=True)
    pack.add_argument("--retry-report-output", required=True)
    pack.add_argument("--retry-report-csv", required=True)
    pack.add_argument("--candidate-pack-output", required=True)

    decision = sub.add_parser("decision-memo")
    decision.add_argument("--stage1-json", required=True)
    decision.add_argument("--stage2-json", required=True)
    decision.add_argument("--stage3-json", required=True)
    decision.add_argument("--stage4-json", required=True)
    decision.add_argument("--gate-json", required=True)
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
    if args.command == "stage1-report":
        campaign = _load_yaml(Path(args.campaign_config))
        targets = _discover_new_block_targets(campaign, Path(args.stage1_root))
        eval_rows = _evaluate_modes(targets, args.device, args.episodes, REPRO_MODES)
        summary_rows = [_summary_row_from_eval(rows) for rows in _group_eval_rows(eval_rows).values()]
        _render_stage1(summary_rows, Path(args.output), Path(args.csv), Path(args.json))
        return
    if args.command == "stage2-report":
        campaign = _load_yaml(Path(args.campaign_config))
        historical_rows = _historical_candidate_rows(Path(campaign["historical_candidate_csv"]), campaign["current_candidate_name"])
        _render_stage2(
            campaign["current_candidate_name"],
            historical_rows,
            _read_csv_rows(Path(args.stage1_csv)),
            Path(args.output),
            Path(args.csv),
            Path(args.json),
        )
        return
    if args.command == "stage3-report":
        campaign = _load_yaml(Path(args.campaign_config))
        _render_stage3(campaign, _read_csv_rows(Path(args.route_csv)), Path(args.output), Path(args.csv), Path(args.json))
        return
    if args.command == "stage4-report":
        campaign = _load_yaml(Path(args.campaign_config))
        stage1_payload = _read_json(Path(args.stage1_json))
        targets, cases = _build_stability_targets(campaign, stage1_payload)
        rows = _evaluate_targets(targets, args.device, args.episodes)
        _render_stage4(rows, cases, Path(args.output), Path(args.csv), Path(args.json))
        return
    if args.command == "successor-draft":
        campaign = _load_yaml(Path(args.campaign_config))
        _render_successor_draft(
            campaign,
            _read_json(Path(args.stage2_json)),
            _read_json(Path(args.stage3_json)),
            _read_json(Path(args.stage4_json)),
            Path(args.output),
            Path(args.json),
        )
        return
    if args.command == "candidate-pack":
        campaign = _load_yaml(Path(args.campaign_config))
        combined_rows, retry_rows, expanded_rows = _candidate_pack_rows(
            Path(campaign["frozen_combined_csv"]),
            Path(campaign["historical_candidate_csv"]),
            Path(args.stage1_csv),
        )
        _render_candidate_pack(
            campaign,
            combined_rows,
            retry_rows,
            expanded_rows,
            Path(args.summary_output),
            Path(args.metrics_output),
            Path(args.combined_report_output),
            Path(args.combined_report_csv),
            Path(args.retry_report_output),
            Path(args.retry_report_csv),
            Path(args.candidate_pack_output),
        )
        return
    if args.command == "decision-memo":
        gate_payload = _read_json(Path(args.gate_json))
        _render_decision_memo(
            _read_json(Path(args.stage1_json)),
            _read_json(Path(args.stage2_json)),
            _read_json(Path(args.stage3_json)),
            _read_json(Path(args.stage4_json)),
            gate_payload,
            Path(args.output),
        )
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
