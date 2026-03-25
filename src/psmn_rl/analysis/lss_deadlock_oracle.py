from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import yaml

from psmn_rl.analysis.campaign_config import load_campaign_config
from psmn_rl.analysis.lss_deadlock_program import (
    _analysis_label,
    _label_run_dir,
    _load_summary,
    _optional_json,
    _round6_run_dir,
    _summary_metrics,
)
from psmn_rl.analysis.lss_frozen_claim import _evaluate_modes
from psmn_rl.analysis.lss_robustness import EvalTarget, _format_float
from psmn_rl.utils.io import get_git_commit, get_git_dirty


ORACLE_STAGE_KEYS: tuple[str, ...] = ("teacher_target", "transition_coverage", "combined")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _gpu_count(device: str) -> int:
    if device == "cpu":
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _student_config_path(
    campaign: dict[str, Any],
    *,
    lane: str,
    seed: int,
    meta: dict[str, Any],
    spec_path: Path,
) -> Path:
    student = campaign["students"]["sare"]
    student_root = Path(campaign["lane_roots"][lane]["sare_student_root"]) / f"seed_{seed}"
    base_config = student_root / "configs" / student["config_name"]
    overrides = meta.get("student_config_overrides", {})
    if not overrides:
        return base_config
    rendered = spec_path.parent / f"{student['output_label']}_student_config.yaml"
    payload = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    payload = _deep_merge(payload, dict(overrides))
    rendered.parent.mkdir(parents=True, exist_ok=True)
    rendered.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return rendered


def _render_oracle_spec(
    campaign: dict[str, Any],
    *,
    candidate: str,
    lane: str,
    seed: int,
    spec_path: Path,
    output_dir: Path,
) -> None:
    meta = dict(campaign["oracle_candidates"][candidate])
    raw = yaml.safe_load(Path(meta["template"]).read_text(encoding="utf-8")) or {}
    raw = _deep_merge(raw, dict(meta.get("overrides", {})))
    student = campaign["students"]["sare"]
    teacher_seed_root = Path(campaign["lane_roots"][lane]["teacher_root"]) / f"seed_{seed}"
    student_seed_root = Path(campaign["lane_roots"][lane]["sare_student_root"]) / f"seed_{seed}"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    raw["name"] = f"{candidate}_{lane}_seed_{seed}_{student['output_label']}"
    raw["output_dir"] = str(output_dir)
    raw["teacher"]["config"] = str(teacher_seed_root / "configs" / "flat_dense_ent1e3.yaml")
    raw["teacher"]["checkpoint"] = str(teacher_seed_root / "flat_dense_ent1e3" / "latest.pt")
    raw["student"]["config"] = str(_student_config_path(campaign, lane=lane, seed=seed, meta=meta, spec_path=spec_path))
    raw["student"]["checkpoint"] = str(student_seed_root / student["run_name"] / "latest.pt")
    spec_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _spec_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _oracle_cases(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = campaign["analysis"]["oracle_case_groups"]
    for subgroup in campaign["analysis"]["oracle_case_order"]:
        for lane, seed in groups[str(subgroup)]:
            rows.append({"subgroup": str(subgroup), "lane": str(lane), "seed": int(seed)})
    return rows


def _oracle_stage_root(campaign: dict[str, Any], stage_key: str) -> Path:
    return Path(campaign["stage_roots"][f"oracle_{stage_key}"])


def _oracle_manifest_path(output_dir: Path) -> Path:
    return output_dir / "oracle_manifest.json"


def _task_output_dir(campaign: dict[str, Any], stage_key: str, candidate: str, lane: str, seed: int) -> Path:
    return _oracle_stage_root(campaign, stage_key) / candidate / lane / f"seed_{seed}" / "kl_lss_sare"


def _task_spec_path(campaign: dict[str, Any], stage_key: str, candidate: str, lane: str, seed: int) -> Path:
    return _oracle_stage_root(campaign, stage_key) / candidate / lane / f"seed_{seed}" / "configs" / "kl_lss_sare.yaml"


def _write_oracle_manifest(task: dict[str, Any], *, status: str, gpu_slot: int | None = None, exit_code: int | None = None) -> None:
    manifest_path = Path(task["manifest_path"])
    payload = dict(task)
    payload["status"] = status
    if gpu_slot is not None:
        payload["gpu_slot"] = gpu_slot
    if exit_code is not None:
        payload["exit_code"] = exit_code
    payload["updated_at"] = _timestamp()
    _write_json(manifest_path, payload)


def _build_oracle_tasks(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for stage_key in ORACLE_STAGE_KEYS:
        root = _oracle_stage_root(campaign, stage_key)
        root.mkdir(parents=True, exist_ok=True)
        for candidate in campaign["analysis"]["oracle_runs"][stage_key]["candidates"]:
            meta = campaign["oracle_candidates"][str(candidate)]
            for case in _oracle_cases(campaign):
                lane = str(case["lane"])
                seed = int(case["seed"])
                output_dir = _task_output_dir(campaign, stage_key, str(candidate), lane, seed)
                spec_path = _task_spec_path(campaign, stage_key, str(candidate), lane, seed)
                _render_oracle_spec(
                    campaign,
                    candidate=str(candidate),
                    lane=lane,
                    seed=seed,
                    spec_path=spec_path,
                    output_dir=output_dir,
                )
                task = {
                    "candidate": str(candidate),
                    "oracle_stage": stage_key,
                    "subgroup": str(case["subgroup"]),
                    "lane": lane,
                    "seed": seed,
                    "config_path": str(spec_path),
                    "config_hash": _spec_hash(spec_path),
                    "seed_block": f"{lane}:{seed}",
                    "family_label": str(meta.get("oracle_stage", stage_key)),
                    "output_root": str(output_dir),
                    "rerun_lineage": "fresh",
                    "manifest_path": str(_oracle_manifest_path(output_dir)),
                    "start_timestamp": None,
                    "end_timestamp": None,
                    "git_commit": get_git_commit(),
                    "git_dirty": get_git_dirty(),
                }
                _write_oracle_manifest(task, status="pending")
                tasks.append(task)
    return tasks


def _launch_oracle_task(task: dict[str, Any], *, device: str, gpu_slot: int | None) -> subprocess.Popen[str]:
    output_dir = Path(task["output_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "psmn_rl.analysis.learner_state_supervision",
        "run",
        "--spec",
        str(task["config_path"]),
        "--device",
        "cuda" if gpu_slot is not None and device != "cpu" else device,
    ]
    (output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    task["start_timestamp"] = _timestamp()
    payload = dict(task)
    _write_oracle_manifest(payload, status="running", gpu_slot=gpu_slot)
    env = os.environ.copy()
    if gpu_slot is not None and device != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_slot)
    return subprocess.Popen(command, cwd=Path.cwd(), env=env, text=True)


def run_oracles(campaign: dict[str, Any], *, device: str) -> None:
    tasks = _build_oracle_tasks(campaign)
    queue_manifest = {
        "program_label": _analysis_label(campaign, "Oracle Program"),
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "device_request": device,
        "visible_gpu_count": _gpu_count(device),
        "queued_tasks": tasks,
        "created_at": _timestamp(),
    }
    _write_json(Path(campaign["reports"]["oracle_queue_manifest_json"]), queue_manifest)
    pending = [task for task in tasks if not (Path(task["output_root"]) / "summary.json").exists()]
    if not pending:
        return
    count = _gpu_count(device)
    if device == "cpu" or count <= 0:
        for task in pending:
            proc = _launch_oracle_task(task, device="cpu", gpu_slot=None)
            exit_code = proc.wait()
            payload = dict(task)
            payload["end_timestamp"] = _timestamp()
            payload["start_timestamp"] = payload["start_timestamp"] or _timestamp()
            _write_oracle_manifest(payload, status="completed" if exit_code == 0 else "failed", exit_code=exit_code)
            if exit_code != 0:
                raise RuntimeError(f"oracle task failed: {task['candidate']} {task['lane']}:{task['seed']}")
        return

    active: list[tuple[subprocess.Popen[str], dict[str, Any], int]] = []
    failed: list[str] = []

    def wait_one() -> None:
        while True:
            for index, (proc, task, slot) in enumerate(active):
                exit_code = proc.poll()
                if exit_code is None:
                    continue
                active.pop(index)
                payload = dict(task)
                payload["end_timestamp"] = _timestamp()
                payload["start_timestamp"] = payload["start_timestamp"] or _timestamp()
                _write_oracle_manifest(payload, status="completed" if exit_code == 0 else "failed", gpu_slot=slot, exit_code=exit_code)
                if exit_code != 0:
                    failed.append(f"{task['candidate']} {task['lane']}:{task['seed']}")
                return
            time.sleep(1.0)

    slot = 0
    for task in pending:
        gpu_slot = slot % count
        proc = _launch_oracle_task(task, device=device, gpu_slot=gpu_slot)
        active.append((proc, task, gpu_slot))
        slot += 1
        if len(active) >= count:
            wait_one()
    while active:
        wait_one()
    if failed:
        raise RuntimeError(f"oracle tasks failed: {failed}")


def _oracle_stage_rows(campaign: dict[str, Any], stage_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in _oracle_cases(campaign):
        lane = str(case["lane"])
        seed = int(case["seed"])
        subgroup = str(case["subgroup"])
        for candidate in campaign["analysis"]["oracle_runs"][stage_key]["candidates"]:
            summary = _load_summary(_task_output_dir(campaign, stage_key, str(candidate), lane, seed))
            metrics = _summary_metrics(summary)
            rows.append(
                {
                    "candidate": str(candidate),
                    "oracle_stage": stage_key,
                    "subgroup": subgroup,
                    "lane": lane,
                    "seed": seed,
                    "final_success": float(summary.get("final_greedy_success", 0.0)),
                    "best_success": float(summary.get("best_round_greedy_success", 0.0)),
                    "first_correct_round": metrics.get("first_correct_round"),
                    "first_stable_round": metrics.get("first_stable_round"),
                    "post_correct_curve_changes": metrics.get("post_correct_curve_changes"),
                    "final_disagreement": float(metrics.get("final_disagreement", 0.0)),
                    "final_teacher_confidence": float(summary["rounds"][-1].get("collection/teacher_confidence_mean", 0.0)),
                    "final_unique_state_ratio": float(summary["rounds"][-1].get("collection/unique_state_ratio", 0.0)),
                    "final_carry_key_frac": float(summary["rounds"][-1].get("collection/phase_frac_carry_key", 0.0)),
                    "final_locked_door_frac": float(summary["rounds"][-1].get("collection/phase_frac_at_locked_door", 0.0)),
                    "final_post_unlock_frac": float(summary["rounds"][-1].get("collection/phase_frac_post_unlock", 0.0)),
                }
            )
    return rows


def _aggregate_oracle_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row["candidate"]), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for candidate, candidate_rows in sorted(by_candidate.items()):
        def subgroup_mean(name: str, key: str) -> float:
            values = [float(row[key]) for row in candidate_rows if str(row["subgroup"]) == name]
            return mean(values) if values else 0.0

        aggregates.append(
            {
                "candidate": candidate,
                "overall_mean": mean(float(row["final_success"]) for row in candidate_rows),
                "teacher_locked_mean": subgroup_mean("teacher_locked", "final_success"),
                "ambiguous_mean": subgroup_mean("ambiguous", "final_success"),
                "guardrail_mean": subgroup_mean("guardrail", "final_success"),
                "teacher_locked_best_mean": subgroup_mean("teacher_locked", "best_success"),
                "carry_key_mean": mean(float(row["final_carry_key_frac"]) for row in candidate_rows),
                "locked_door_mean": mean(float(row["final_locked_door_frac"]) for row in candidate_rows),
                "post_unlock_mean": mean(float(row["final_post_unlock_frac"]) for row in candidate_rows),
                "unique_state_ratio_mean": mean(float(row["final_unique_state_ratio"]) for row in candidate_rows),
                "mean_first_stable_round": mean(
                    float(row["first_stable_round"]) for row in candidate_rows if row["first_stable_round"] is not None
                )
                if any(row["first_stable_round"] is not None for row in candidate_rows)
                else None,
                "zero_final_cases": sum(1 for row in candidate_rows if float(row["final_success"]) <= 0.0),
            }
        )
    return aggregates


def _best_aggregate(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        aggregates,
        key=lambda row: (
            float(row["teacher_locked_mean"]),
            float(row["overall_mean"]),
            float(row["guardrail_mean"]),
            float(row["carry_key_mean"]),
        ),
    )


def _oracle_signal(best: dict[str, Any], baseline: dict[str, Any]) -> str:
    locked_gain = float(best["teacher_locked_mean"]) - float(baseline["teacher_locked_mean"])
    overall_gain = float(best["overall_mean"]) - float(baseline["overall_mean"])
    if locked_gain >= 0.15 or overall_gain >= 0.10:
        return "meaningful_headroom"
    if locked_gain >= 0.05 or overall_gain >= 0.05:
        return "partial_signal"
    return "no_actionable_signal"


def _render_oracle_stage_report(
    campaign: dict[str, Any],
    *,
    stage_key: str,
    output: Path,
    json_output: Path | None,
    title: str,
    question_lines: list[str],
) -> dict[str, Any]:
    rows = _oracle_stage_rows(campaign, stage_key)
    aggregates = _aggregate_oracle_rows(rows)
    baseline_name = str(campaign["analysis"]["oracle_runs"][stage_key]["baseline_candidate"])
    baseline = next(row for row in aggregates if str(row["candidate"]) == baseline_name)
    best = _best_aggregate(aggregates)
    signal = _oracle_signal(best, baseline)
    lines = [
        f"# {title}",
        "",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        f"- baseline oracle candidate: `{baseline_name}`",
        f"- best oracle candidate: `{best['candidate']}`",
        f"- mechanism signal: `{signal}`",
        "",
    ]
    lines.extend(question_lines)
    lines.extend(
        [
            "",
            "| Candidate | Overall Mean | Teacher-Locked Mean | Ambiguous Mean | Guardrail Mean | Carry-Key | Locked-Door | Post-Unlock | Zero-Final Cases |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(aggregates, key=lambda item: (item["teacher_locked_mean"], item["overall_mean"]), reverse=True):
        lines.append(
            f"| `{row['candidate']}` | `{row['overall_mean']:.4f}` | `{row['teacher_locked_mean']:.4f}` | `{row['ambiguous_mean']:.4f}` | `{row['guardrail_mean']:.4f}` | `{row['carry_key_mean']:.4f}` | `{row['locked_door_mean']:.4f}` | `{row['post_unlock_mean']:.4f}` | `{int(row['zero_final_cases'])}` |"
        )
    lines.extend(
        [
            "",
            "| Candidate | Subgroup | Lane | Seed | Final Success | Best Success | First Stable Round | Unique-State Ratio | Carry-Key | Locked-Door | Post-Unlock |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["candidate"], item["subgroup"], item["lane"], item["seed"])):
        lines.append(
            f"| `{row['candidate']}` | `{row['subgroup']}` | {row['lane']} | {row['seed']} | `{row['final_success']:.4f}` | `{row['best_success']:.4f}` | `{_format_float(row['first_stable_round'])}` | `{row['final_unique_state_ratio']:.4f}` | `{row['final_carry_key_frac']:.4f}` | `{row['final_locked_door_frac']:.4f}` | `{row['final_post_unlock_frac']:.4f}` |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "rows": rows,
        "aggregate_rows": aggregates,
        "baseline_candidate": baseline_name,
        "best_candidate": best["candidate"],
        "signal": signal,
        "best": best,
        "baseline": baseline,
    }
    if json_output is not None:
        _write_json(json_output, payload)
    return payload


def render_teacher_target_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    _render_oracle_stage_report(
        campaign,
        stage_key="teacher_target",
        output=output,
        json_output=json_output,
        title="Oracle Stage A2 Teacher-Target Oracle",
        question_lines=[
            "## Question",
            "",
            "- Does correcting the teacher target move the canonical teacher-locked subgroup?",
            "- Or does the student stay stuck even when the target is softened or confidence-clipped?",
        ],
    )


def render_transition_coverage_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    _render_oracle_stage_report(
        campaign,
        stage_key="transition_coverage",
        output=output,
        json_output=json_output,
        title="Oracle Stage A3 Transition-Coverage Oracle",
        question_lines=[
            "## Question",
            "",
            "- Does correcting transition-state coverage move the deadlock family?",
            "- Or does the model still fail even when carry-key and locked-door coverage are pushed directly?",
        ],
    )


def render_combined_report(campaign: dict[str, Any], output: Path, json_output: Path | None) -> None:
    teacher_payload = _optional_json(campaign["reports"].get("oracle_teacher_target_json"))
    transition_payload = _optional_json(campaign["reports"].get("oracle_transition_json"))
    payload = _render_oracle_stage_report(
        campaign,
        stage_key="combined",
        output=output,
        json_output=json_output,
        title="Oracle Stage A4 Combined Oracle",
        question_lines=[
            "## Question",
            "",
            "- Is the mixed barrier breakable only when teacher-target repair and transition-coverage repair are applied together?",
            "- Or does a strong residual ceiling remain even under the combined oracle?",
        ],
    )
    if not json_output:
        return
    combined = _read_json(json_output)
    teacher_best = dict(teacher_payload.get("best", {}))
    transition_best = dict(transition_payload.get("best", {}))
    best = dict(payload["best"])
    baseline = dict(payload["baseline"])
    best_individual = max(
        [teacher_best, transition_best],
        key=lambda row: (float(row.get("teacher_locked_mean", 0.0)), float(row.get("overall_mean", 0.0))),
    ) if teacher_best or transition_best else {}
    combined_gain = float(best["teacher_locked_mean"]) - float(best_individual.get("teacher_locked_mean", baseline["teacher_locked_mean"]))
    combined_payload = dict(combined)
    combined_payload["best_individual_oracle"] = best_individual
    combined_payload["combined_vs_best_individual_teacher_locked_gain"] = combined_gain
    combined_payload["combined_classification"] = (
        "combined_headroom"
        if combined_gain >= 0.10 or float(best["overall_mean"]) >= float(best_individual.get("overall_mean", baseline["overall_mean"])) + 0.10
        else "residual_ceiling"
    )
    _write_json(json_output, combined_payload)


def _oracle_eval_target(label: str, run_dir: Path, lane: str, seed: int) -> EvalTarget:
    config_path = run_dir / "student_resolved_config.yaml"
    if not config_path.exists():
        config_path = run_dir / "resolved_config.yaml"
    return EvalTarget(
        seed=seed,
        label=label,
        variant=label,
        config_path=config_path,
        checkpoint_path=run_dir / "latest.pt",
        run_dir=run_dir,
        method=label,
        stage=lane,
        metadata={"lane": lane},
        command_path=run_dir / "command.txt",
    )


def render_inference_escape_report(
    campaign: dict[str, Any],
    output: Path,
    csv_output: Path | None,
    json_output: Path | None,
    *,
    device: str,
) -> None:
    combined_payload = _optional_json(campaign["reports"].get("oracle_combined_json"))
    combined_best = str(combined_payload.get("best_candidate") or campaign["analysis"]["oracle_runs"]["combined"]["candidates"][0])
    modes = [
        (str(name), bool(greedy), float(temp))
        for name, greedy, temp in campaign["analysis"]["oracle_eval_modes"]
    ]
    targets: list[EvalTarget] = []
    for case in _oracle_cases(campaign):
        lane = str(case["lane"])
        seed = int(case["seed"])
        targets.append(_oracle_eval_target("round6", _round6_run_dir(campaign, lane, seed), lane, seed))
        combined_dir = _task_output_dir(campaign, "combined", combined_best, lane, seed)
        if (combined_dir / "summary.json").exists():
            targets.append(_oracle_eval_target(combined_best, combined_dir, lane, seed))
    rows = _evaluate_modes(targets, device, int(campaign["analysis"]["oracle_eval_episodes"]), modes)
    aggregates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    subgroup_lookup = {(str(case["lane"]), int(case["seed"])): str(case["subgroup"]) for case in _oracle_cases(campaign)}
    for row in rows:
        subgroup = subgroup_lookup[(str(row["lane"]), int(row["seed"]))]
        row["subgroup"] = subgroup
        aggregates.setdefault((str(row["label"]), subgroup), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (label, subgroup), grouped in sorted(aggregates.items()):
        greedy = next(item for item in grouped if str(item["mode"]) == "greedy")
        best_sampled = max(
            [item for item in grouped if str(item["mode"]) != "greedy"],
            key=lambda item: float(item.get("eval_success_rate", 0.0)),
            default=greedy,
        )
        summary_rows.append(
            {
                "label": label,
                "subgroup": subgroup,
                "greedy_success": float(greedy.get("eval_success_rate", 0.0)),
                "best_sampled_success": float(best_sampled.get("eval_success_rate", 0.0)),
                "best_sampled_mode": str(best_sampled.get("mode", "greedy")),
                "sampled_gain": float(best_sampled.get("eval_success_rate", 0.0)) - float(greedy.get("eval_success_rate", 0.0)),
            }
        )
    teacher_locked_gain = max(
        (float(row["sampled_gain"]) for row in summary_rows if row["label"] == "round6" and row["subgroup"] == "teacher_locked"),
        default=0.0,
    )
    inference_classification = (
        "cheap_escape_signal"
        if teacher_locked_gain >= 0.10
        else "weak_or_subgroup_only_signal"
        if max((float(row["sampled_gain"]) for row in summary_rows if row["label"] == "round6"), default=0.0) >= 0.05
        else "no_actionable_inference_signal"
    )
    lines = [
        "# Oracle Stage A5 Inference Escape Oracle",
        "",
        f"- evaluated incumbent: `round6`",
        f"- evaluated combined oracle line: `{combined_best}`",
        f"- inference classification: `{inference_classification}`",
        "",
        "## Question",
        "",
        "- Can the canonical deadlock be escaped cheaply at inference time?",
        "- Or is the problem still upstream in supervision or representation?",
        "",
        "| Label | Subgroup | Greedy Success | Best Sampled Success | Best Sampled Mode | Sampled Gain |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for row in sorted(summary_rows, key=lambda item: (item["label"], item["subgroup"])):
        lines.append(
            f"| `{row['label']}` | `{row['subgroup']}` | `{row['greedy_success']:.4f}` | `{row['best_sampled_success']:.4f}` | `{row['best_sampled_mode']}` | `{row['sampled_gain']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "| Label | Lane | Seed | Mode | Eval Success |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["label"]), str(item["lane"]), int(item["seed"]), str(item["mode"]))):
        lines.append(
            f"| `{row['label']}` | {row['lane']} | {row['seed']} | `{row['mode']}` | `{float(row.get('eval_success_rate', 0.0)):.4f}` |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_output is not None:
        _write_csv(csv_output, rows)
    if json_output is not None:
        _write_json(
            json_output,
            {
                "rows": rows,
                "summary_rows": summary_rows,
                "combined_best": combined_best,
                "inference_classification": inference_classification,
            },
        )


def render_synthesis(
    campaign: dict[str, Any],
    output: Path,
    shortlist_output: Path,
    json_output: Path | None,
    shortlist_json_output: Path | None,
) -> None:
    teacher = _optional_json(campaign["reports"].get("oracle_teacher_target_json"))
    transition = _optional_json(campaign["reports"].get("oracle_transition_json"))
    combined = _optional_json(campaign["reports"].get("oracle_combined_json"))
    inference = _optional_json(campaign["reports"].get("oracle_inference_json"))

    teacher_signal = str(teacher.get("signal", "missing"))
    transition_signal = str(transition.get("signal", "missing"))
    combined_signal = str(combined.get("combined_classification", combined.get("signal", "missing")))
    inference_signal = str(inference.get("inference_classification", "missing"))

    ranking_rows = [
        {
            "mechanism_class": "teacher_target_problem",
            "signal": teacher_signal,
            "best_candidate": teacher.get("best_candidate"),
            "teacher_locked_mean": teacher.get("best", {}).get("teacher_locked_mean", 0.0),
            "overall_mean": teacher.get("best", {}).get("overall_mean", 0.0),
        },
        {
            "mechanism_class": "transition_coverage_problem",
            "signal": transition_signal,
            "best_candidate": transition.get("best_candidate"),
            "teacher_locked_mean": transition.get("best", {}).get("teacher_locked_mean", 0.0),
            "overall_mean": transition.get("best", {}).get("overall_mean", 0.0),
        },
        {
            "mechanism_class": "mixed_teacher_plus_coverage_problem",
            "signal": combined_signal,
            "best_candidate": combined.get("best_candidate"),
            "teacher_locked_mean": combined.get("best", {}).get("teacher_locked_mean", 0.0),
            "overall_mean": combined.get("best", {}).get("overall_mean", 0.0),
        },
        {
            "mechanism_class": "inference_time_escape_problem",
            "signal": inference_signal,
            "best_candidate": inference.get("combined_best"),
            "teacher_locked_mean": max(
                (float(row["sampled_gain"]) for row in inference.get("summary_rows", []) if row["subgroup"] == "teacher_locked"),
                default=0.0,
            ),
            "overall_mean": max((float(row["sampled_gain"]) for row in inference.get("summary_rows", [])), default=0.0),
        },
    ]

    architecture_branch_justified = (
        combined_signal == "residual_ceiling"
        and teacher_signal == "no_actionable_signal"
        and transition_signal == "no_actionable_signal"
        and inference_signal == "no_actionable_inference_signal"
    )
    mechanism_verdict = (
        "oracle evidence points to a likely family ceiling"
        if architecture_branch_justified
        else "mixed problem with practical in-family headroom"
        if combined_signal == "combined_headroom" or teacher_signal == "meaningful_headroom" or transition_signal == "meaningful_headroom"
        else "mixed problem remains unresolved inside the current family"
    )

    shortlist_rows = [
        row
        for row in campaign["analysis"]["practical_shortlist"]
        if str(row["track"]) != "archpilot" or architecture_branch_justified
    ][: int(campaign["analysis"]["shortlist_max_directions"])]

    lines = [
        "# Oracle Stage A6 Synthesis",
        "",
        f"- mechanism verdict: `{mechanism_verdict}`",
        f"- architecture branch justified: `{architecture_branch_justified}`",
        "",
        "| Mechanism Class | Signal | Best Candidate | Teacher-Locked Signal | Overall Signal |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in ranking_rows:
        lines.append(
            f"| `{row['mechanism_class']}` | `{row['signal']}` | `{row['best_candidate']}` | `{float(row['teacher_locked_mean']):.4f}` | `{float(row['overall_mean']):.4f}` |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    shortlist_lines = [
        "# Oracle Practical Shortlist",
        "",
        f"- shortlisted practical directions: `{len(shortlist_rows)}`",
        f"- shortlist ceiling: `{campaign['analysis']['shortlist_max_directions']}`",
        f"- architecture branch justified: `{architecture_branch_justified}`",
        "",
    ]
    for index, row in enumerate(shortlist_rows, start=1):
        shortlist_lines.extend(
            [
                f"## Direction {index}: {row['label']}",
                "",
                f"- track: `{row['track']}`",
                f"- candidate set: `{row['candidates']}`",
                f"- mechanism hypothesis: {row['hypothesis']}",
                f"- failure signature: {row['failure']}",
                "",
            ]
        )
    shortlist_output.parent.mkdir(parents=True, exist_ok=True)
    shortlist_output.write_text("\n".join(shortlist_lines) + "\n", encoding="utf-8")

    payload = {
        "ranking_rows": ranking_rows,
        "mechanism_verdict": mechanism_verdict,
        "architecture_branch_justified": architecture_branch_justified,
    }
    shortlist_payload = {
        "practical_directions": shortlist_rows,
        "architecture_branch_justified": architecture_branch_justified,
        "mechanism_verdict": mechanism_verdict,
    }
    if json_output is not None:
        _write_json(json_output, payload)
    if shortlist_json_output is not None:
        _write_json(shortlist_json_output, shortlist_payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Oracle-driven deadlock helpers around the active round6 benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-oracles")
    run.add_argument("--campaign-config", required=True)
    run.add_argument("--device", default="auto")

    for name in ("teacher-target-report", "transition-coverage-report", "combined-report"):
        item = sub.add_parser(name)
        item.add_argument("--campaign-config", required=True)
        item.add_argument("--output", required=True)
        item.add_argument("--json", required=False)

    inference = sub.add_parser("inference-escape-report")
    inference.add_argument("--campaign-config", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--csv", required=False)
    inference.add_argument("--json", required=False)
    inference.add_argument("--device", default="auto")

    synthesis = sub.add_parser("synthesis-report")
    synthesis.add_argument("--campaign-config", required=True)
    synthesis.add_argument("--output", required=True)
    synthesis.add_argument("--shortlist-output", required=True)
    synthesis.add_argument("--json", required=False)
    synthesis.add_argument("--shortlist-json", required=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign = load_campaign_config(Path(args.campaign_config))
    if args.command == "run-oracles":
        run_oracles(campaign, device=str(args.device))
        return
    if args.command == "teacher-target-report":
        render_teacher_target_report(campaign, Path(args.output), Path(args.json) if args.json else None)
        return
    if args.command == "transition-coverage-report":
        render_transition_coverage_report(campaign, Path(args.output), Path(args.json) if args.json else None)
        return
    if args.command == "combined-report":
        render_combined_report(campaign, Path(args.output), Path(args.json) if args.json else None)
        return
    if args.command == "inference-escape-report":
        render_inference_escape_report(
            campaign,
            Path(args.output),
            Path(args.csv) if args.csv else None,
            Path(args.json) if args.json else None,
            device=str(args.device),
        )
        return
    if args.command == "synthesis-report":
        render_synthesis(
            campaign,
            Path(args.output),
            Path(args.shortlist_output),
            Path(args.json) if args.json else None,
            Path(args.shortlist_json) if args.shortlist_json else None,
        )
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
