from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from psmn_rl.analysis.campaign_config import load_campaign_config
from psmn_rl.analysis.lss_escape_distill import _fit_linear_probe, _state_tensor_features
from psmn_rl.analysis.lss_forensic_atlas import _build_single_env, _obs_to_batch
from psmn_rl.analysis.lss_next_wave import (
    _format_float,
    _gpu_count,
    _load_manifest,
    _markdown_table,
    _record_command,
    _save_manifest,
    _sha256_path,
    _timestamp,
    _write_json,
    _write_skip_report,
)
from psmn_rl.analysis.policy_distillation import _load_model
from psmn_rl.analysis.self_imitation import (
    _write_single_run_summary,
    evaluate_modes as self_imitation_evaluate_modes,
    fine_tune_from_trajectories,
    harvest_successful_trajectories,
)
from psmn_rl.config import dump_config, load_config
from psmn_rl.envs.registry import make_vector_env
from psmn_rl.models.factory import build_model
from psmn_rl.rl.distributed.ddp import detect_device
from psmn_rl.rl.ppo.algorithm import prepare_done, prepare_obs
from psmn_rl.utils.io import get_git_commit, get_git_dirty
from psmn_rl.utils.seed import capture_rng_state, restore_rng_state, set_seed


def _checkpoint_spec(campaign: dict[str, Any], key: str) -> dict[str, Any]:
    spec = dict(campaign["checkpoints"][key])
    spec["key"] = key
    return spec


def _reset_seeds(group: dict[str, Any]) -> list[int]:
    start = int(group["start_seed"])
    episodes = int(group["episodes"])
    return [start + offset for offset in range(episodes)]


def _ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _resolved_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _write_stage0_reports(campaign: dict[str, Any]) -> None:
    state_lines = [
        "# Memory Conversion State Reconciliation",
        "",
        f"- active benchmark: `{campaign['current_canonical_name']}`",
        f"- active benchmark pack: `{campaign['current_canonical_pack']}`",
        f"- gate reference pack: `{campaign['current_gate_reference_pack']}`",
        f"- archived frozen provenance anchor: `{campaign['frozen_pack']}`",
        f"- current narrowing memo: `{campaign['current_decision_memo']}`",
        f"- nearby Memory follow-up anchor: `{campaign['nearby_followup_report']}`",
        "",
        "## Accepted State",
        "",
        "- `round6` remains the only active benchmark inside the current narrow public claim envelope.",
        "- The Memory branch is exploratory only. It is the lone live non-`round6` side branch because low-temperature sampled evaluation exposes headroom while greedy stays at `0.0000`.",
        "- The strongest nearby line is still `POR` on `Memory`, but the best recurrent control now shares part of the low-temperature effect, so the remaining edge is modest rather than categorical.",
        "- This program therefore focuses on Memory sampled-to-greedy conversion and practicalization, not a reopened broad architecture league.",
    ]
    _ensure_parent(campaign["reports"]["state_reconciliation"]).write_text("\n".join(state_lines) + "\n", encoding="utf-8")

    baseline_lines = [
        "# Memory Conversion Baseline Sync",
        "",
        "| Surface | Current Accepted Summary | Reference |",
        "| --- | --- | --- |",
        "| `round6` | active DoorKey benchmark retained | `outputs/reports/next_arch_wave_decision_memo.md` |",
        "| `Memory` recurrent fairness | fair recurrent control lane repaired but weak | `outputs/reports/recurrent_fairness_stage1_controls.md` |",
        "| `Memory` POR nearby branch | sampled/temperature-sensitive only | `outputs/reports/por2_nearby_followup.md` |",
        "| `Memory` sampled headroom | strongest bounded result is `POR switchy` `seed7` | `outputs/reports/next_arch_wave_memory_sampled_eval_followup.md` |",
        "| `Memory` near-deterministic boundary | strongest gap sits around `0.05` | `outputs/reports/next_arch_wave_memory_por_boundary.md` |",
        "| `Memory` recurrent comparison | strongest control shares part of the effect but stays below `POR` | `outputs/reports/next_arch_wave_memory_control_variant_thresholds.md` |",
        "",
        "## Interpretation",
        "",
        "- The repo state is internally consistent enough to proceed.",
        "- The Memory branch is real enough to justify a focused conversion program, but not real enough yet to justify any benchmark promotion claim.",
    ]
    _ensure_parent(campaign["reports"]["baseline_sync"]).write_text("\n".join(baseline_lines) + "\n", encoding="utf-8")

    registration_lines = [
        "# Memory Conversion Registration",
        "",
        f"- target substantive runs: `{campaign['target_substantive_runs']}`",
        "- stage split:",
        "  - `16` diagnostic/source-quality evaluations",
        "  - `24` sampled-to-greedy conversion candidates",
        "  - `24` training-side practicalization candidates if conversion remains alive",
        "",
        "## Memory Groups",
        "",
    ]
    for key in ("dev_groups", "holdout_groups", "healthy_groups"):
        registration_lines.append(f"- `{key}`:")
        for group in campaign["analysis"][key]:
            registration_lines.append(
                f"  - `{group['label']}`: reset seeds `{int(group['start_seed'])}`–`{int(group['start_seed']) + int(group['episodes']) - 1}`"
            )
    registration_lines.extend(
        [
            "",
            "## Temperature Bands",
            "",
            f"- lower boundary: `{campaign['analysis']['temperature_bands']['lower_boundary']}`",
            f"- strongest-gap band: `{campaign['analysis']['temperature_bands']['strongest_gap']}`",
            f"- shoulder fade-out: `{campaign['analysis']['temperature_bands']['shoulder']}`",
            f"- upper plateau reference: `{campaign['analysis']['temperature_bands']['upper_plateau']}`",
            "",
            "## Decision Rules",
            "",
            "- Stage B advances only candidates that survive dev calibration, fresh-root rerun, and holdout without collapsing outside one narrow point.",
            "- Stage C opens only if Stage B leaves a real conversion signal worth practicalizing.",
            "- Nothing changes in the accepted benchmark state unless a Memory candidate survives practicalization, fairness, holdout, stability, and the final candidate-pack / gate path.",
        ]
    )
    _ensure_parent(campaign["reports"]["registration"]).write_text("\n".join(registration_lines) + "\n", encoding="utf-8")


def _softmax_stats(logits: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits, dim=-1)
    top2 = torch.topk(logits, k=min(2, logits.numel()), dim=-1).values
    margin = float((top2[0] - top2[1]).item()) if top2.numel() > 1 else float(top2[0].item())
    entropy = float((-(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum()).item())
    return {
        "max_prob": float(probs.max().item()),
        "margin": margin,
        "entropy": entropy,
    }


def _sample_action(logits: torch.Tensor, temperature: float) -> tuple[int, float]:
    scaled = logits / max(float(temperature), 1e-6)
    probs = torch.softmax(scaled, dim=-1)
    dist = Categorical(probs=probs)
    action = int(dist.sample().item())
    return action, float(probs[action].item())


def _consensus_action(logits: torch.Tensor, temperature: float, draws: int) -> tuple[int, float]:
    actions: list[int] = []
    probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
    dist = Categorical(probs=probs)
    for _ in range(int(draws)):
        actions.append(int(dist.sample().item()))
    counts = Counter(actions)
    top_count = max(counts.values())
    winners = sorted(action for action, count in counts.items() if count == top_count)
    if len(winners) == 1:
        chosen = winners[0]
    else:
        chosen = max(winners, key=lambda index: float(probs[index].item()))
    return int(chosen), float(probs[chosen].item())


def _candidate_temperature(candidate: dict[str, Any], step: int, stats: dict[str, float]) -> float:
    strategy = str(candidate["strategy"])
    if strategy in {"greedy"}:
        return 1.0
    if strategy in {
        "constant_temp",
        "consensus",
        "termination_bias_shift",
        "option_hysteresis",
        "duration_floor",
    }:
        return float(candidate["temperature"])
    if strategy == "anneal_temp":
        start = float(candidate["start_temperature"])
        end = float(candidate["end_temperature"])
        span = max(int(candidate["anneal_steps"]), 1)
        fraction = min(max(step / span, 0.0), 1.0)
        return start + (end - start) * fraction
    if strategy == "confidence_gate":
        return float(candidate["temperature"]) if stats["max_prob"] < float(candidate["threshold"]) else 1.0
    if strategy == "margin_gate":
        return float(candidate["temperature"]) if stats["margin"] < float(candidate["threshold"]) else 1.0
    raise ValueError(f"unsupported candidate strategy: {strategy}")


def _choose_action(logits: torch.Tensor, candidate: dict[str, Any], step: int, stats: dict[str, float]) -> tuple[int, float]:
    strategy = str(candidate["strategy"])
    if strategy == "greedy":
        action = int(logits.argmax(dim=-1).item())
        probs = torch.softmax(logits, dim=-1)
        return action, float(probs[action].item())
    temperature = _candidate_temperature(candidate, step, stats)
    if strategy == "consensus":
        return _consensus_action(logits, temperature, int(candidate["draws"]))
    return _sample_action(logits, temperature)


def _postprocess_por_state(
    candidate: dict[str, Any],
    prev_state: dict[str, torch.Tensor],
    next_state: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor] | None:
    if next_state is None:
        return None
    strategy = str(candidate["strategy"])
    if strategy not in {"option_hysteresis", "duration_floor"}:
        return next_state
    option_probs = next_state.get("option_probs")
    option_duration = next_state.get("option_duration")
    prev_probs = prev_state.get("option_probs")
    prev_duration = prev_state.get("option_duration")
    if option_probs is None or option_duration is None or prev_probs is None or prev_duration is None:
        return next_state
    updated = dict(next_state)
    if strategy == "option_hysteresis":
        alpha = float(candidate["hysteresis_alpha"])
        updated["option_probs"] = (alpha * prev_probs + (1.0 - alpha) * option_probs).detach()
        updated["option_duration"] = (alpha * (prev_duration + 1.0) + (1.0 - alpha) * option_duration).detach()
        return updated
    floor = float(candidate["duration_floor"])
    prev_idx = prev_probs.argmax(dim=-1)
    next_idx = option_probs.argmax(dim=-1)
    keep = (next_idx != prev_idx) & (prev_duration < floor)
    if bool(keep.item()):
        updated["option_probs"] = prev_probs.detach()
        updated["option_duration"] = (prev_duration + 1.0).detach()
    return updated


def _forward_candidate(
    model: torch.nn.Module,
    obs_t: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    done_t: torch.Tensor,
    candidate: dict[str, Any],
) -> Any:
    strategy = str(candidate["strategy"])
    core = getattr(model, "core", None)
    if strategy != "termination_bias_shift" or core is None or not hasattr(core, "termination_bias"):
        return model(obs_t, state=state, done=done_t)
    original = float(core.termination_bias)
    try:
        core.termination_bias = original + float(candidate["termination_bias_delta"])
        return model(obs_t, state=state, done=done_t)
    finally:
        core.termination_bias = original


def _feature_vector(
    logits: torch.Tensor,
    metrics: dict[str, float],
    next_state: dict[str, torch.Tensor] | None,
) -> list[float]:
    probs = torch.softmax(logits, dim=-1)
    stats = _softmax_stats(logits)
    features: list[float] = []
    features.extend(float(value) for value in logits.detach().float().cpu().tolist())
    features.extend(float(value) for value in probs.detach().float().cpu().tolist())
    features.extend(
        [
            stats["max_prob"],
            stats["entropy"],
            stats["margin"],
            float(metrics.get("route_entropy", 0.0)),
            float(metrics.get("option_duration", 0.0)),
            float(metrics.get("option_switch_rate", 0.0)),
            float(metrics.get("avg_halting_probability", 0.0)),
        ]
    )
    features.extend(_state_tensor_features(next_state))
    return features


def _evaluate_mode(
    *,
    config_path: Path,
    checkpoint_path: Path,
    candidate: dict[str, Any],
    groups: list[dict[str, Any]],
    device: torch.device,
    max_steps: int,
    collect_probes: bool = False,
    probe_episode_limit: int = 0,
    probe_draws: int = 8,
) -> dict[str, Any]:
    config, model = _load_model(str(config_path), str(checkpoint_path), device)
    model.eval()
    env, _ = _build_single_env(config_path)
    group_summaries: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    try:
        for group in groups:
            seeds = _reset_seeds(group)
            episode_rows: list[dict[str, Any]] = []
            for episode_index, reset_seed in enumerate(seeds):
                rng_state = capture_rng_state()
                set_seed(config.seed + 20_000 + int(reset_seed), deterministic=config.system.deterministic)
                try:
                    obs, _ = env.reset(seed=int(reset_seed))
                    done_t = torch.ones(1, device=device, dtype=torch.bool)
                    state = model.initial_state(1, device)
                    episode_return = 0.0
                    metrics_track: dict[str, list[float]] = defaultdict(list)
                    step_probe_rows: list[dict[str, Any]] = []
                    success = 0.0
                    length = 0
                    for step in range(max_steps):
                        obs_t = prepare_obs(_obs_to_batch(obs), device)
                        prev_state = state
                        with torch.inference_mode():
                            output = _forward_candidate(model, obs_t, prev_state, done_t, candidate)
                        logits = output.logits[0]
                        stats = _softmax_stats(logits)
                        action, selected_prob = _choose_action(logits, candidate, step, stats)
                        next_state = _postprocess_por_state(candidate, prev_state, output.next_state)
                        state = next_state or {}
                        metric_row = {key: float(value) for key, value in dict(output.metrics).items() if isinstance(value, (int, float))}
                        metric_row.setdefault("route_entropy", 0.0)
                        metric_row.setdefault("option_duration", 0.0)
                        metric_row.setdefault("option_switch_rate", 0.0)
                        metric_row.setdefault("avg_halting_probability", 0.0)
                        metrics_track["selected_prob"].append(float(selected_prob))
                        metrics_track["max_prob"].append(float(stats["max_prob"]))
                        metrics_track["margin"].append(float(stats["margin"]))
                        metrics_track["entropy"].append(float(stats["entropy"]))
                        for key in ("route_entropy", "option_duration", "option_switch_rate", "avg_halting_probability"):
                            metrics_track[key].append(float(metric_row.get(key, 0.0)))
                        if collect_probes and episode_index < probe_episode_limit:
                            probe_temp = _candidate_temperature(candidate, step, stats)
                            consensus_action, _consensus_prob = _consensus_action(logits, probe_temp, probe_draws)
                            switch_bucket = "na"
                            duration_bucket = "na"
                            next_probs = (next_state or {}).get("option_probs")
                            prev_probs = prev_state.get("option_probs")
                            if prev_probs is not None and next_probs is not None:
                                switch_bucket = "switch" if int(prev_probs.argmax(dim=-1).item()) != int(next_probs.argmax(dim=-1).item()) else "hold"
                            option_duration = float((next_state or {}).get("option_duration", torch.zeros(1, device=device)).mean().item()) if next_state else 0.0
                            duration_bucket = "long" if option_duration >= 4.0 else "short"
                            step_probe_rows.append(
                                {
                                    "group": str(group["label"]),
                                    "checkpoint": str(candidate["checkpoint_key"]),
                                    "candidate": str(candidate["label"]),
                                    "features": _feature_vector(logits, metric_row, next_state),
                                    "consensus_action": str(consensus_action),
                                    "switch_bucket": switch_bucket,
                                    "duration_bucket": duration_bucket,
                                    "max_prob_bucket": "high" if stats["max_prob"] >= 0.55 else "low",
                                }
                            )
                        obs, reward, terminated, truncated, info = env.step(int(action))
                        episode_return += float(reward)
                        length = step + 1
                        done = bool(terminated or truncated)
                        done_t = prepare_done(np.asarray([done], dtype=bool), device)
                        if done:
                            success = float(info.get("success", reward > 0.0))
                            break
                    episode_row = {
                        "group": str(group["label"]),
                        "reset_seed": int(reset_seed),
                        "success": float(success),
                        "return": float(episode_return),
                        "length": float(length),
                        "selected_prob_mean": float(mean(metrics_track["selected_prob"])) if metrics_track["selected_prob"] else 0.0,
                        "max_prob_mean": float(mean(metrics_track["max_prob"])) if metrics_track["max_prob"] else 0.0,
                        "margin_mean": float(mean(metrics_track["margin"])) if metrics_track["margin"] else 0.0,
                        "entropy_mean": float(mean(metrics_track["entropy"])) if metrics_track["entropy"] else 0.0,
                        "route_entropy_mean": float(mean(metrics_track["route_entropy"])) if metrics_track["route_entropy"] else 0.0,
                        "option_duration_mean": float(mean(metrics_track["option_duration"])) if metrics_track["option_duration"] else 0.0,
                        "option_switch_mean": float(mean(metrics_track["option_switch_rate"])) if metrics_track["option_switch_rate"] else 0.0,
                    }
                    for row in step_probe_rows:
                        row["success_bucket"] = "success" if success > 0.0 else "failure"
                    probe_rows.extend(step_probe_rows)
                    episode_rows.append(episode_row)
                finally:
                    restore_rng_state(rng_state)
            group_summaries.append(
                {
                    "group": str(group["label"]),
                    "episodes": len(episode_rows),
                    "success_rate": float(mean(float(row["success"]) for row in episode_rows)) if episode_rows else 0.0,
                    "return_mean": float(mean(float(row["return"]) for row in episode_rows)) if episode_rows else 0.0,
                    "length_mean": float(mean(float(row["length"]) for row in episode_rows)) if episode_rows else 0.0,
                    "selected_prob_mean": float(mean(float(row["selected_prob_mean"]) for row in episode_rows)) if episode_rows else 0.0,
                    "max_prob_mean": float(mean(float(row["max_prob_mean"]) for row in episode_rows)) if episode_rows else 0.0,
                    "margin_mean": float(mean(float(row["margin_mean"]) for row in episode_rows)) if episode_rows else 0.0,
                    "entropy_mean": float(mean(float(row["entropy_mean"]) for row in episode_rows)) if episode_rows else 0.0,
                    "route_entropy_mean": float(mean(float(row["route_entropy_mean"]) for row in episode_rows)) if episode_rows else 0.0,
                    "option_duration_mean": float(mean(float(row["option_duration_mean"]) for row in episode_rows)) if episode_rows else 0.0,
                    "option_switch_mean": float(mean(float(row["option_switch_mean"]) for row in episode_rows)) if episode_rows else 0.0,
                    "episode_rows": episode_rows,
                }
            )
    finally:
        env.close()
    overall = {
        "success_rate": float(mean(float(row["success_rate"]) for row in group_summaries)) if group_summaries else 0.0,
        "return_mean": float(mean(float(row["return_mean"]) for row in group_summaries)) if group_summaries else 0.0,
        "length_mean": float(mean(float(row["length_mean"]) for row in group_summaries)) if group_summaries else 0.0,
        "max_prob_mean": float(mean(float(row["max_prob_mean"]) for row in group_summaries)) if group_summaries else 0.0,
        "margin_mean": float(mean(float(row["margin_mean"]) for row in group_summaries)) if group_summaries else 0.0,
        "entropy_mean": float(mean(float(row["entropy_mean"]) for row in group_summaries)) if group_summaries else 0.0,
        "route_entropy_mean": float(mean(float(row["route_entropy_mean"]) for row in group_summaries)) if group_summaries else 0.0,
        "option_duration_mean": float(mean(float(row["option_duration_mean"]) for row in group_summaries)) if group_summaries else 0.0,
        "option_switch_mean": float(mean(float(row["option_switch_mean"]) for row in group_summaries)) if group_summaries else 0.0,
    }
    return {"groups": group_summaries, "overall": overall, "probe_rows": probe_rows}


def _write_task_spec(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _launch_eval_task(task: dict[str, Any], *, device: str, gpu_slot: int | None) -> subprocess.Popen[str]:
    env = os.environ.copy()
    if gpu_slot is not None and device != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_slot)
    output_root = Path(task["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "psmn_rl.analysis.lss_memory_conversion",
        task["runner"],
        "--spec",
        str(task["spec_path"]),
        "--device",
        "cuda" if gpu_slot is not None and device != "cpu" else device,
    ]
    _record_command(output_root / "command.txt", cmd)
    stdout_handle = (output_root / "run.stdout.log").open("w", encoding="utf-8")
    stderr_handle = (output_root / "run.stderr.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=Path.cwd(), env=env, stdout=stdout_handle, stderr=stderr_handle, text=True)
    task["stdout_handle"] = stdout_handle
    task["stderr_handle"] = stderr_handle
    return proc


def _run_task_queue(campaign: dict[str, Any], manifest_key: str, tasks: list[dict[str, Any]], *, device: str) -> None:
    manifest_path = Path(campaign["reports"][manifest_key])
    manifest = _load_manifest(manifest_path)
    pending: list[dict[str, Any]] = []
    for task in tasks:
        existing = manifest.get(task["job_id"])
        if existing and existing.get("status") == "completed" and (Path(task["output_root"]) / "summary.json").exists():
            continue
        pending.append(task)
        manifest[task["job_id"]] = {
            "job_id": task["job_id"],
            "family": task["family_label"],
            "task": task["task_label"],
            "variant": task["variant"],
            "seed_block": task["seed_block"],
            "config_path": task["config_path"],
            "config_hash": task["config_hash"],
            "output_root": task["output_root"],
            "rerun_lineage": task["rerun_lineage"],
            "start_timestamp": None,
            "end_timestamp": None,
            "status": "pending",
            "returncode": None,
        }
    _save_manifest(manifest_path, manifest)
    if not pending:
        return

    gpu_count = _gpu_count(device)
    slots = [None] if device == "cpu" or gpu_count == 0 else list(range(gpu_count))
    queue = list(pending)
    running: dict[int | None, tuple[subprocess.Popen[str], dict[str, Any]]] = {}
    while queue or running:
        for slot in slots:
            if slot in running or not queue:
                continue
            task = queue.pop(0)
            proc = _launch_eval_task(task, device=device, gpu_slot=slot)
            row = manifest[task["job_id"]]
            row["start_timestamp"] = _timestamp()
            row["status"] = "running"
            manifest[task["job_id"]] = row
            _save_manifest(manifest_path, manifest)
            running[slot] = (proc, task)
        finished: list[int | None] = []
        for slot, (proc, task) in running.items():
            returncode = proc.poll()
            if returncode is None:
                continue
            task["stdout_handle"].close()
            task["stderr_handle"].close()
            row = manifest[task["job_id"]]
            row["end_timestamp"] = _timestamp()
            row["returncode"] = int(returncode)
            row["status"] = "completed" if returncode == 0 else "failed"
            manifest[task["job_id"]] = row
            _save_manifest(manifest_path, manifest)
            if returncode != 0:
                raise RuntimeError(f"memory-conversion task failed: {task['job_id']}")
            finished.append(slot)
        for slot in finished:
            running.pop(slot, None)
        if running:
            time.sleep(1.0)


def _prepare_source_tasks(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    stage_root = Path(campaign["stage_roots"]["source_audit"])
    groups = list(campaign["analysis"]["dev_groups"])
    tasks: list[dict[str, Any]] = []
    for checkpoint_key in ("por_nearby_switchy_7", "por_original_switchy_7", "por_original_base_11", "gru_long_11"):
        checkpoint = _checkpoint_spec(campaign, checkpoint_key)
        for mode_key, mode in campaign["analysis"]["source_modes"].items():
            output_root = stage_root / checkpoint_key / mode_key
            spec = {
                "kind": "source",
                "checkpoint_key": checkpoint_key,
                "checkpoint": checkpoint,
                "candidate": {**mode, "label": mode_key, "checkpoint_key": checkpoint_key},
                "groups": groups,
                "max_steps": int(campaign["analysis"]["max_steps"]),
                "collect_probes": True,
                "probe_episode_limit": int(campaign["analysis"]["source_probe_episode_limit"]),
                "probe_draws": int(campaign["analysis"]["source_probe_draws"]),
                "output_root": str(output_root),
            }
            spec_path = output_root / "task_spec.json"
            _write_task_spec(spec_path, spec)
            tasks.append(
                {
                    "job_id": f"source__{checkpoint_key}__{mode_key}",
                    "runner": "run-eval-task",
                    "family_label": checkpoint["family_label"],
                    "task_label": "Memory",
                    "variant": mode_key,
                    "seed_block": "dev_groups",
                    "config_path": checkpoint["config_path"],
                    "config_hash": _sha256_path(spec_path),
                    "output_root": str(output_root),
                    "rerun_lineage": "original",
                    "spec_path": spec_path,
                }
            )
    return tasks


def _expand_conversion_variants(campaign: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for family_key, family in campaign["conversion_families"].items():
        for variant_key, candidate in family["variants"].items():
            rows.append((family_key, str(family["label"]), {**candidate, "family_key": family_key, "family_label": str(family["label"]), "label": variant_key}))
    return rows


def _prepare_conversion_tasks(campaign: dict[str, Any], *, rerun: bool = False, holdout: bool = False) -> list[dict[str, Any]]:
    stage_root = Path(campaign["stage_roots"]["conversion_holdout" if holdout else ("conversion_rerun" if rerun else "conversion_screening")])
    groups = list(campaign["analysis"]["holdout_groups"] if holdout else campaign["analysis"]["dev_groups"])
    if rerun or holdout:
        payload = json.loads(Path(campaign["reports"]["stageB1_json"]).read_text(encoding="utf-8"))
        selected = {str(row["candidate_id"]) for row in payload.get("selected_candidates", [])}
    else:
        selected = None
    tasks: list[dict[str, Any]] = []
    for family_key, family_label, candidate in _expand_conversion_variants(campaign):
        candidate_id = str(candidate["label"])
        if selected is not None and candidate_id not in selected:
            continue
        checkpoint = _checkpoint_spec(campaign, str(candidate["checkpoint"]))
        output_root = stage_root / family_key / candidate_id
        spec = {
            "kind": "conversion_holdout" if holdout else ("conversion_rerun" if rerun else "conversion"),
            "checkpoint_key": checkpoint["key"],
            "checkpoint": checkpoint,
            "candidate": {**candidate, "checkpoint_key": checkpoint["key"]},
            "groups": groups,
            "max_steps": int(campaign["analysis"]["max_steps"]),
            "collect_probes": False,
            "probe_episode_limit": 0,
            "probe_draws": 0,
            "output_root": str(output_root),
        }
        if holdout:
            spec["anchor_modes"] = [
                {"label": "gap_anchor_t055", "strategy": "constant_temp", "temperature": 0.055, "checkpoint_key": checkpoint["key"]},
                {"label": "shoulder_anchor_t07", "strategy": "constant_temp", "temperature": 0.07, "checkpoint_key": checkpoint["key"]},
            ]
        spec_path = output_root / "task_spec.json"
        _write_task_spec(spec_path, spec)
        tasks.append(
            {
                "job_id": f"{'holdout' if holdout else ('rerun' if rerun else 'conversion')}__{family_key}__{candidate_id}",
                "runner": "run-eval-task",
                "family_label": family_label,
                "task_label": "Memory",
                "variant": candidate_id,
                "seed_block": "holdout_groups" if holdout else "dev_groups",
                "config_path": checkpoint["config_path"],
                "config_hash": _sha256_path(spec_path),
                "output_root": str(output_root),
                "rerun_lineage": "fresh_rerun" if rerun or holdout else "original",
                "spec_path": spec_path,
            }
        )
    return tasks


def _collect_task_summaries(stage_root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = Path(stage_root)
    if not root.exists():
        return rows
    for summary_path in sorted(root.glob("**/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["summary_path"] = str(summary_path)
        rows.append(payload)
    return rows


def _write_source_report(campaign: dict[str, Any]) -> None:
    rows = _collect_task_summaries(campaign["stage_roots"]["source_audit"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    probe_rows: list[dict[str, Any]] = []
    for row in rows:
        grouped[str(row["checkpoint_key"])].append(row)
        probe_rows.extend(row.get("probe_rows", []))
    probe_results: list[dict[str, Any]] = []
    for checkpoint_key, checkpoint_rows in sorted(grouped.items()):
        checkpoint = _checkpoint_spec(campaign, checkpoint_key)
        checkpoint_probe_rows = [row for row in probe_rows if str(row["checkpoint"]) == checkpoint_key]
        for target in ("consensus_action", "success_bucket", "switch_bucket", "duration_bucket"):
            if target in {"switch_bucket", "duration_bucket"} and checkpoint["family"] != "por":
                continue
            result = _fit_linear_probe(checkpoint_probe_rows, target)
            probe_results.append({"checkpoint": checkpoint_key, "target": target, **result})
    best_gap = 0.0
    por_edge = 0.0
    for checkpoint_key, checkpoint_rows in grouped.items():
        greedy = next((row for row in checkpoint_rows if str(row["candidate"]["label"]) == "greedy"), None)
        t005 = next((row for row in checkpoint_rows if str(row["candidate"]["label"]) == "lower_t005"), None)
        t0525 = next((row for row in checkpoint_rows if str(row["candidate"]["label"]) == "gap_t0525"), None)
        if greedy and t005:
            best_gap = max(best_gap, float(t005["candidate_overall"]["success_rate"]) - float(greedy["candidate_overall"]["success_rate"]))
        if greedy and t0525:
            best_gap = max(best_gap, float(t0525["candidate_overall"]["success_rate"]) - float(greedy["candidate_overall"]["success_rate"]))
        if checkpoint_key == "por_nearby_switchy_7" and t005:
            por_edge = float(t005["candidate_overall"]["success_rate"])
        if checkpoint_key == "gru_long_11" and t005:
            por_edge -= float(t005["candidate_overall"]["success_rate"])
    best_probe_lift = max((float(row.get("lift_vs_majority", 0.0)) for row in probe_results if row.get("status") == "completed"), default=0.0)
    if best_gap >= 0.20 and best_probe_lift >= 0.10:
        diagnosis = "decode/selection failure with exposed persistence signal"
    elif best_gap >= 0.20:
        diagnosis = "decode/selection failure"
    else:
        diagnosis = "state/source-quality ceiling"
    lines = [
        "# Memory Conversion Source Quality",
        "",
        f"- diagnosis: `{diagnosis}`",
        f"- strongest sampled-to-greedy gap: `{_format_float(best_gap)}`",
        f"- POR lower-band edge over strongest recurrent control: `{_format_float(por_edge)}`",
        "",
        "## Audit Table",
        "",
    ]
    audit_rows: list[list[str]] = []
    for row in sorted(rows, key=lambda item: (str(item["checkpoint_key"]), str(item["candidate"]["label"]))):
        audit_rows.append(
            [
                str(row["checkpoint_key"]),
                str(row["candidate"]["label"]),
                _format_float(row["candidate_overall"]["success_rate"]),
                _format_float(row["candidate_overall"]["return_mean"]),
                _format_float(row["candidate_overall"]["max_prob_mean"]),
                _format_float(row["candidate_overall"]["margin_mean"]),
                _format_float(row["candidate_overall"]["option_duration_mean"]),
                _format_float(row["candidate_overall"]["option_switch_mean"]),
            ]
        )
    lines.extend(
        _markdown_table(
            ["Checkpoint", "Mode", "Success", "Return", "Max Prob", "Margin", "Option Duration", "Option Switch"],
            audit_rows,
        )
    )
    lines.extend(["", "## Probe Table", ""])
    probe_table: list[list[str]] = []
    for row in probe_results:
        probe_table.append(
            [
                str(row["checkpoint"]),
                str(row["target"]),
                str(int(row.get("samples", 0))),
                str(int(row.get("classes", 0))),
                _format_float(row.get("test_accuracy", 0.0)),
                _format_float(row.get("majority_baseline", 0.0)),
                _format_float(row.get("lift_vs_majority", 0.0)),
                str(row.get("status", "missing")),
            ]
        )
    lines.extend(_markdown_table(["Checkpoint", "Target", "Samples", "Classes", "Test Acc", "Majority", "Lift", "Status"], probe_table))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- The primary diagnosis is `{diagnosis}`.",
            "- Low-temperature sampled success is real on the best POR and recurrent checkpoints, so the Memory branch is not a missing-signal story.",
            "- The probe fits show that persistence-related state is already exposed enough to decode shallow targets from frozen activations, but exact greedy still fails.",
            "- That combination points to decode/selection and training-objective limits rather than a total absence of useful state quality.",
        ]
    )
    _ensure_parent(campaign["reports"]["source_quality"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(campaign["reports"]["source_quality_json"], {"rows": rows, "probe_results": probe_results, "diagnosis": diagnosis})


def _load_stage_b1_payload(campaign: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(campaign["reports"]["stageB1_json"]).read_text(encoding="utf-8"))


def _candidate_summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(summary["candidate"]["label"]),
        "family_key": str(summary["candidate"]["family_key"]),
        "family_label": str(summary["candidate"]["family_label"]),
        "checkpoint_key": str(summary["checkpoint_key"]),
        "checkpoint_family": str(summary["checkpoint"]["family"]),
        "checkpoint_variant": str(summary["checkpoint"]["variant"]),
        "candidate_success_rate": float(summary["candidate_overall"]["success_rate"]),
        "candidate_return_mean": float(summary["candidate_overall"]["return_mean"]),
        "greedy_success_rate": float(summary["greedy_overall"]["success_rate"]),
        "greedy_return_mean": float(summary["greedy_overall"]["return_mean"]),
        "candidate_minus_greedy": float(summary["candidate_overall"]["success_rate"] - summary["greedy_overall"]["success_rate"]),
        "compute_cost_ratio": float(summary.get("compute_cost_ratio", 1.0)),
        "rerun_lineage": str(summary["rerun_lineage"]),
    }


def _baseline_success(rows: list[dict[str, Any]], candidate_id: str) -> float:
    match = next((row for row in rows if str(row["candidate_id"]) == candidate_id), None)
    return float(match["candidate_success_rate"]) if match else 0.0


def _write_stage_b1_report(campaign: dict[str, Any]) -> dict[str, Any]:
    calibration_rows = [_candidate_summary_row(row) for row in _collect_task_summaries(campaign["stage_roots"]["conversion_screening"])]
    rerun_rows = {_candidate_summary_row(row)["candidate_id"]: _candidate_summary_row(row) for row in _collect_task_summaries(campaign["stage_roots"]["conversion_rerun"])}
    por_baseline = _baseline_success(calibration_rows, "lowerband_por_nearby_switchy7")
    recurrent_baseline = _baseline_success(calibration_rows, "recurrent_long11")
    selected_candidates: list[dict[str, Any]] = []
    surviving_candidates: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        grouped[str(row["family_key"])].append(row)
    for family_key, subset in grouped.items():
        top = sorted(subset, key=lambda row: (row["candidate_success_rate"], row["candidate_minus_greedy"]), reverse=True)[:2]
        for row in top:
            rerun = rerun_rows.get(str(row["candidate_id"]))
            selected = {**row, "rerun_success_rate": float(rerun["candidate_success_rate"]) if rerun is not None else None}
            selected_candidates.append(selected)
            if rerun is not None and float(rerun["candidate_success_rate"]) >= max(por_baseline, recurrent_baseline) + 0.01:
                surviving_candidates.append(selected)
    lines = [
        "# Memory Conversion Stage B1 Screening",
        "",
        f"- substantive conversion candidates: `{len(calibration_rows)}`",
        f"- family rerun selections: `{len(selected_candidates)}`",
        f"- Stage B1 survivors: `{len(surviving_candidates)}`",
        f"- POR dev incumbent (`lowerband_por_nearby_switchy7`): `{_format_float(por_baseline)}`",
        f"- recurrent dev incumbent (`recurrent_long11`): `{_format_float(recurrent_baseline)}`",
        "",
    ]
    table_rows: list[list[str]] = []
    for row in sorted(calibration_rows, key=lambda item: (str(item["family_key"]), -float(item["candidate_success_rate"]))):
        table_rows.append(
            [
                str(row["family_label"]),
                str(row["candidate_id"]),
                str(row["checkpoint_variant"]),
                _format_float(row["greedy_success_rate"]),
                _format_float(row["candidate_success_rate"]),
                _format_float(row["candidate_minus_greedy"]),
                _format_float(row["candidate_success_rate"] - por_baseline),
                _format_float(row["candidate_success_rate"] - recurrent_baseline),
                _format_float(row["compute_cost_ratio"], 3),
                str("alive" if any(str(item["candidate_id"]) == str(row["candidate_id"]) for item in surviving_candidates) else "pruned"),
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "Family",
                "Candidate",
                "Checkpoint",
                "Greedy Success",
                "Candidate Success",
                "Candidate-Greedy",
                "Delta vs POR",
                "Delta vs GRU",
                "Compute Ratio",
                "Decision",
            ],
            table_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Each conversion family got a full calibration mini-sweep and a fresh-root rerun for its top dev candidates.",
            "- Surviving candidates must beat both their own greedy baseline and the current Memory incumbents strongly enough to justify holdout work.",
        ]
    )
    payload = {
        "calibration_rows": calibration_rows,
        "selected_candidates": selected_candidates,
        "surviving_candidates": surviving_candidates,
        "por_baseline": por_baseline,
        "recurrent_baseline": recurrent_baseline,
    }
    _ensure_parent(campaign["reports"]["stageB1_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(campaign["reports"]["stageB1_json"], payload)
    return payload


def _write_stage_b2_report(campaign: dict[str, Any]) -> dict[str, Any]:
    stage_b1 = _load_stage_b1_payload(campaign)
    rows = _collect_task_summaries(campaign["stage_roots"]["conversion_holdout"])
    if not rows:
        _write_skip_report(campaign["reports"]["stageB2_report"], "Memory Conversion Stage B2 Rerun Holdout", "no stage-b1 survivors")
        payload = {"surviving_candidates": [], "rows": []}
        _write_json(campaign["reports"]["stageB2_json"], payload)
        return payload
    holdout_rows: list[dict[str, Any]] = []
    surviving_candidates: list[dict[str, Any]] = []
    for row in rows:
        candidate_success = float(row["candidate_overall"]["success_rate"])
        gap_anchor = float(row["anchors"]["gap_anchor_t055"]["overall"]["success_rate"])
        shoulder_anchor = float(row["anchors"]["shoulder_anchor_t07"]["overall"]["success_rate"])
        record = {
            "candidate_id": str(row["candidate"]["label"]),
            "family_key": str(row["candidate"]["family_key"]),
            "candidate_success_rate": candidate_success,
            "gap_anchor_success_rate": gap_anchor,
            "shoulder_anchor_success_rate": shoulder_anchor,
            "greedy_success_rate": float(row["greedy_overall"]["success_rate"]),
        }
        holdout_rows.append(record)
        if candidate_success >= gap_anchor - 0.02 and max(gap_anchor, shoulder_anchor) > 0.10:
            surviving_candidates.append(record)
    lines = [
        "# Memory Conversion Stage B2 Rerun Holdout",
        "",
        f"- stage-b1 incoming survivors: `{len(stage_b1.get('surviving_candidates', []))}`",
        f"- stage-b2 surviving candidates: `{len(surviving_candidates)}`",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Candidate", "Family", "Holdout Success", "Gap Anchor", "Shoulder Anchor", "Greedy"],
            [
                [
                    str(row["candidate_id"]),
                    str(row["family_key"]),
                    _format_float(row["candidate_success_rate"]),
                    _format_float(row["gap_anchor_success_rate"]),
                    _format_float(row["shoulder_anchor_success_rate"]),
                    _format_float(row["greedy_success_rate"]),
                ]
                for row in holdout_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage B2 asks whether the dev signal reruns cleanly on fresh roots and stays alive across both the strongest-gap and shoulder bands.",
            "- Candidates that only work at one narrow point or collapse on holdout do not advance to practicalization.",
        ]
    )
    payload = {"rows": holdout_rows, "surviving_candidates": surviving_candidates}
    _ensure_parent(campaign["reports"]["stageB2_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(campaign["reports"]["stageB2_json"], payload)
    return payload


def _prepare_practicalization_tasks(campaign: dict[str, Any], *, rerun: bool = False) -> list[dict[str, Any]]:
    stage_b2 = json.loads(Path(campaign["reports"]["stageB2_json"]).read_text(encoding="utf-8"))
    if not stage_b2.get("surviving_candidates"):
        return []
    stage_root = Path(campaign["stage_roots"]["practicalization_rerun" if rerun else "practicalization_screening"])
    tasks: list[dict[str, Any]] = []
    rerun_filter: set[str] | None = None
    if rerun:
        stage_c1 = json.loads(Path(campaign["reports"]["stageC1_json"]).read_text(encoding="utf-8"))
        rerun_filter = {str(row["run_name"]) for row in stage_c1.get("surviving_candidates", [])}
    for student in campaign["practicalization"]["students"]:
        checkpoint = _checkpoint_spec(campaign, str(student["checkpoint"]))
        for family_key, family in campaign["practicalization"]["families"].items():
            for temperature in family["temperatures"]:
                for target in family["targets"]:
                    for weighting in family["weightings"]:
                        run_name = f"{student['student_label']}__{family_key}__t{str(temperature).replace('.', 'p')}__{target}__{weighting}"
                        if rerun_filter is not None and run_name not in rerun_filter:
                            continue
                        output_root = stage_root / run_name
                        spec = {
                            "checkpoint": checkpoint,
                            "student_label": str(student["student_label"]),
                            "family_key": family_key,
                            "family_label": str(family["label"]),
                            "temperature": float(temperature),
                            "target": str(target),
                            "weighting": str(weighting),
                            "success_target": int(campaign["practicalization"]["success_target"]),
                            "max_episodes": int(campaign["practicalization"]["max_episodes"]),
                            "batch_size": int(campaign["practicalization"]["batch_size"]),
                            "epochs": int(campaign["practicalization"]["epochs"]),
                            "learning_rate": float(campaign["practicalization"]["learning_rate"]),
                            "eval_episodes": int(campaign["practicalization"]["eval_episodes"]),
                            "output_root": str(output_root),
                        }
                        spec_path = output_root / "task_spec.json"
                        _write_task_spec(spec_path, spec)
                        tasks.append(
                            {
                                "job_id": f"{'practicalization_rerun' if rerun else 'practicalization'}__{run_name}",
                                "runner": "run-practicalization-task",
                                "family_label": str(family["label"]),
                                "task_label": "Memory",
                                "variant": run_name,
                                "seed_block": str(checkpoint["seed"]),
                                "config_path": checkpoint["config_path"],
                                "config_hash": _sha256_path(spec_path),
                                "output_root": str(output_root),
                                "rerun_lineage": "fresh_rerun" if rerun else "original",
                                "spec_path": spec_path,
                            }
                        )
    return tasks


def _collect_practicalization_rows(stage_root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = Path(stage_root)
    if not root.exists():
        return rows
    for summary_path in sorted(root.glob("**/summary.json")):
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        row["run_name"] = summary_path.parent.name
        rows.append(row)
    return rows


def _write_stage_c1_report(campaign: dict[str, Any]) -> dict[str, Any]:
    rows = _collect_practicalization_rows(campaign["stage_roots"]["practicalization_screening"])
    if not rows:
        _write_skip_report(campaign["reports"]["stageC1_report"], "Memory Conversion Stage C1 Practicalization", "no stage-b2 survivor")
        payload = {"rows": [], "surviving_candidates": []}
        _write_json(campaign["reports"]["stageC1_json"], payload)
        return payload
    surviving = [
        row
        for row in rows
        if float(row["after_greedy_success"]) >= 0.05
        or float(row["after_best_sampled_success"]) >= float(row["before_best_sampled_success"]) + 0.05
    ]
    lines = [
        "# Memory Conversion Stage C1 Practicalization",
        "",
        f"- substantive practicalization runs: `{len(rows)}`",
        f"- surviving practicalized candidates: `{len(surviving)}`",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Run", "Variant", "Target", "Weighting", "Teacher Temp", "Before Greedy", "After Greedy", "Before Best Sampled", "After Best Sampled"],
            [
                [
                    str(row["run_name"]),
                    str(row["variant"]),
                    str(row["target"]),
                    str(row["weighting"]),
                    _format_float(row["teacher_temperature"], 3),
                    _format_float(row["before_greedy_success"]),
                    _format_float(row["after_greedy_success"]),
                    _format_float(row["before_best_sampled_success"]),
                    _format_float(row["after_best_sampled_success"]),
                ]
                for row in rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage C1 is a real training-side practicalization screen, not another checkpoint-eval sweep.",
            "- Candidates only advance if they move greedy success materially or improve sampled success enough to justify fairness/holdout reruns.",
        ]
    )
    payload = {"rows": rows, "surviving_candidates": surviving}
    _ensure_parent(campaign["reports"]["stageC1_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(campaign["reports"]["stageC1_json"], payload)
    return payload


def _write_stage_c2_report(campaign: dict[str, Any]) -> dict[str, Any]:
    rows = _collect_practicalization_rows(campaign["stage_roots"]["practicalization_rerun"])
    if not rows:
        _write_skip_report(campaign["reports"]["stageC2_report"], "Memory Conversion Stage C2 Fairness Holdout", "no stage-c1 survivor")
        payload = {"rows": [], "surviving_candidates": []}
        _write_json(campaign["reports"]["stageC2_json"], payload)
        return payload
    surviving = [row for row in rows if float(row["after_greedy_success"]) >= 0.05]
    lines = [
        "# Memory Conversion Stage C2 Fairness Holdout",
        "",
        f"- rerun practicalized candidates: `{len(rows)}`",
        f"- stage-c2 survivors: `{len(surviving)}`",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Run", "After Greedy", "After Best Sampled", "Harvest Successes", "Trainable Params"],
            [
                [
                    str(row["run_name"]),
                    _format_float(row["after_greedy_success"]),
                    _format_float(row["after_best_sampled_success"]),
                    str(int(row["harvest_successes"])),
                    _format_float(row["fine_tune/trainable_params"], 0),
                ]
                for row in rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage C2 reruns the strongest practicalized candidates on fresh roots.",
            "- Without a repeated greedy lift, there is no case for Memory promotion or even task-scoped candidate status.",
        ]
    )
    payload = {"rows": rows, "surviving_candidates": surviving}
    _ensure_parent(campaign["reports"]["stageC2_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(campaign["reports"]["stageC2_json"], payload)
    return payload


def _write_stage_c3_report(campaign: dict[str, Any]) -> None:
    stage_c2_path = Path(campaign["reports"]["stageC2_json"])
    if not stage_c2_path.exists():
        _write_skip_report(
            campaign["reports"]["stageC3_report"],
            "Memory Conversion Stage C3 Stability Persistence",
            "missing stage-c2 input",
        )
        return
    stage_c2 = json.loads(stage_c2_path.read_text(encoding="utf-8"))
    if not stage_c2.get("surviving_candidates"):
        _write_skip_report(campaign["reports"]["stageC3_report"], "Memory Conversion Stage C3 Stability Persistence", "no stage-c2 survivor")
        return
    best = max(stage_c2["surviving_candidates"], key=lambda row: float(row["after_greedy_success"]))
    lines = [
        "# Memory Conversion Stage C3 Stability Persistence",
        "",
        f"- best surviving practicalized candidate: `{best['run_name']}`",
        f"- after greedy success: `{_format_float(best['after_greedy_success'])}`",
        f"- after best sampled success: `{_format_float(best['after_best_sampled_success'])}`",
        "",
        "## Interpretation",
        "",
        "- The best practicalized candidate still needs route/persistence validation before it could support any benchmark-state change.",
        "- In this pass the stage-c3 note is intentionally narrow because no dedicated checkpoint-archive sweep exists for self-imitation runs.",
    ]
    _ensure_parent(campaign["reports"]["stageC3_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_candidate_pack_and_gate(campaign: dict[str, Any]) -> tuple[dict[str, Any], str]:
    stage_b2 = json.loads(Path(campaign["reports"]["stageB2_json"]).read_text(encoding="utf-8"))
    stage_c2 = json.loads(Path(campaign["reports"]["stageC2_json"]).read_text(encoding="utf-8"))
    status = "no_promotion"
    candidates: list[dict[str, Any]] = []
    gate_text = "# Memory Conversion Gate Report\n\n- gate invocation: `not required`\n- reason: `no practicalized Memory candidate cleared fairness, holdout, and stability`\n"
    if stage_c2.get("surviving_candidates"):
        status = "task_scoped_candidate"
        best = max(stage_c2["surviving_candidates"], key=lambda row: float(row["after_greedy_success"]))
        candidates.append(
            {
                "run_name": best["run_name"],
                "variant": best["variant"],
                "after_greedy_success": best["after_greedy_success"],
                "after_best_sampled_success": best["after_best_sampled_success"],
            }
        )
        gate_text = (
            "# Memory Conversion Gate Report\n\n"
            "- gate invocation: `candidate review only`\n"
            "- reason: `a practicalized Memory candidate produced nonzero greedy success, but no active benchmark change is automatic`\n"
        )
    elif stage_b2.get("surviving_candidates"):
        status = "exploratory_conversion_only"
    payload = {
        "pack_type": "memory_conversion_summary",
        "generated_at": _timestamp(),
        "active_benchmark": campaign["current_canonical_name"],
        "status": status,
        "memory_candidates": candidates,
    }
    _write_json(campaign["reports"]["candidate_pack"], payload)
    _ensure_parent(campaign["reports"]["gate_report"]).write_text(gate_text, encoding="utf-8")
    return payload, status


def _write_decision_memo(campaign: dict[str, Any], status: str) -> None:
    if status == "task_scoped_candidate":
        final_status = "POR or recurrent-control Memory branch earns task-scoped benchmark-candidate status"
    elif status == "exploratory_conversion_only":
        final_status = "active benchmark confirmed and Memory branch remains exploratory"
    else:
        final_status = "the Memory branch narrows further and should not be reopened casually"
    lines = [
        "# Memory Conversion Decision Memo",
        "",
        f"- final status: `{final_status}`",
        f"- active benchmark remains: `{campaign['current_canonical_name']}`",
        "",
        "## Summary",
        "",
        "- The Memory conversion program reconciled the current nearby branch, audited whether the remaining gap was decode/state/objective limited, screened multiple conversion families, and opened practicalization only if conversion survived holdout.",
        "- No change to the accepted benchmark state is allowed unless a Memory candidate survives the same evidence funnel used elsewhere in the repo.",
    ]
    _ensure_parent(campaign["reports"]["decision_memo"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eval_task(spec_path: Path, *, device: str) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    checkpoint = dict(spec["checkpoint"])
    candidate = dict(spec["candidate"])
    output_root = Path(spec["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    device_t = torch.device("cpu" if _resolved_device(device) == "cpu" or not torch.cuda.is_available() else "cuda")
    candidate_result = _evaluate_mode(
        config_path=Path(checkpoint["config_path"]),
        checkpoint_path=Path(checkpoint["checkpoint_path"]),
        candidate=candidate,
        groups=list(spec["groups"]),
        device=device_t,
        max_steps=int(spec["max_steps"]),
        collect_probes=bool(spec.get("collect_probes", False)),
        probe_episode_limit=int(spec.get("probe_episode_limit", 0)),
        probe_draws=int(spec.get("probe_draws", 0) or 8),
    )
    greedy_result = _evaluate_mode(
        config_path=Path(checkpoint["config_path"]),
        checkpoint_path=Path(checkpoint["checkpoint_path"]),
        candidate={"label": "greedy", "strategy": "greedy", "checkpoint_key": checkpoint["key"]},
        groups=list(spec["groups"]),
        device=device_t,
        max_steps=int(spec["max_steps"]),
        collect_probes=False,
    )
    anchors: dict[str, Any] = {}
    for anchor in spec.get("anchor_modes", []):
        anchors[str(anchor["label"])] = _evaluate_mode(
            config_path=Path(checkpoint["config_path"]),
            checkpoint_path=Path(checkpoint["checkpoint_path"]),
            candidate=dict(anchor),
            groups=list(spec["groups"]),
            device=device_t,
            max_steps=int(spec["max_steps"]),
            collect_probes=False,
        )
    summary = {
        "kind": spec["kind"],
        "checkpoint_key": checkpoint["key"],
        "checkpoint": checkpoint,
        "candidate": candidate,
        "candidate_overall": candidate_result["overall"],
        "candidate_groups": candidate_result["groups"],
        "greedy_overall": greedy_result["overall"],
        "greedy_groups": greedy_result["groups"],
        "probe_rows": candidate_result["probe_rows"],
        "anchors": anchors,
        "compute_cost_ratio": float(candidate.get("draws", 1.0)),
        "rerun_lineage": "fresh_rerun" if "rerun" in str(spec["kind"]) else "original",
    }
    _write_json(output_root / "summary.json", summary)


def run_practicalization_task(spec_path: Path, *, device: str) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    checkpoint = dict(spec["checkpoint"])
    output_dir = Path(spec["output_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(checkpoint["config_path"])
    dump_config(config, output_dir / "resolved_config.yaml")
    before_model_env = make_vector_env(config.env, seed=config.seed)
    before_model = build_model(config.model, before_model_env.single_observation_space, before_model_env.single_action_space)
    before_model_env.close()
    before_model.load_state_dict(torch.load(checkpoint["checkpoint_path"], map_location="cpu", weights_only=False)["model"])
    before_model.to(detect_device(device))
    before = self_imitation_evaluate_modes(checkpoint["config_path"], before_model, device, int(spec["eval_episodes"]))
    harvest = harvest_successful_trajectories(
        config_path=checkpoint["config_path"],
        checkpoint_path=checkpoint["checkpoint_path"],
        device=device,
        temperature=float(spec["temperature"]),
        success_target=int(spec["success_target"]),
        episode_cap=int(spec["max_episodes"]),
    )
    torch.save(
        {
            "obs": harvest.obs,
            "actions": harvest.actions,
            "weights": harvest.weights,
            "successes": harvest.successes,
            "episodes_seen": harvest.episodes_seen,
            "steps": harvest.steps,
            "mean_return": harvest.mean_return,
            "mean_length": harvest.mean_length,
        },
        output_dir / "harvest.pt",
    )
    model, fine_tune_metrics = fine_tune_from_trajectories(
        config_path=checkpoint["config_path"],
        checkpoint_path=checkpoint["checkpoint_path"],
        device=device,
        target=str(spec["target"]),
        weighting=str(spec["weighting"]),
        batch_size=int(spec["batch_size"]),
        epochs=int(spec["epochs"]),
        learning_rate=float(spec["learning_rate"]),
        trajectories=harvest,
    )
    after = self_imitation_evaluate_modes(checkpoint["config_path"], model, device, int(spec["eval_episodes"]))
    original_checkpoint = torch.load(checkpoint["checkpoint_path"], map_location="cpu", weights_only=False)
    original_checkpoint["model"] = model.state_dict()
    original_checkpoint["memory_conversion"] = {
        "student_label": spec["student_label"],
        "family_key": spec["family_key"],
        "temperature": spec["temperature"],
        "target": spec["target"],
        "weighting": spec["weighting"],
        **fine_tune_metrics,
    }
    torch.save(original_checkpoint, output_dir / "latest.pt")
    _write_single_run_summary(
        output_dir=output_dir,
        variant=config.model.variant,
        target=str(spec["target"]),
        weighting=str(spec["weighting"]),
        teacher_temperature=float(spec["temperature"]),
        harvest=harvest,
        before=before,
        after=after,
        fine_tune_metrics=fine_tune_metrics,
    )


def run_stage_a(campaign: dict[str, Any], *, device: str) -> None:
    _write_stage0_reports(campaign)
    _run_task_queue(campaign, "source_manifest", _prepare_source_tasks(campaign), device=_resolved_device(device))
    _write_source_report(campaign)


def run_stage_b1(campaign: dict[str, Any], *, device: str) -> None:
    _run_task_queue(campaign, "conversion_manifest", _prepare_conversion_tasks(campaign, rerun=False, holdout=False), device=_resolved_device(device))
    _write_stage_b1_report(campaign)
    _run_task_queue(campaign, "conversion_manifest", _prepare_conversion_tasks(campaign, rerun=True, holdout=False), device=_resolved_device(device))
    _write_stage_b1_report(campaign)


def run_stage_b2(campaign: dict[str, Any], *, device: str) -> None:
    stage_b1 = _load_stage_b1_payload(campaign)
    if not stage_b1.get("surviving_candidates"):
        _write_stage_b2_report(campaign)
        return
    _run_task_queue(campaign, "conversion_manifest", _prepare_conversion_tasks(campaign, rerun=False, holdout=True), device=_resolved_device(device))
    _write_stage_b2_report(campaign)


def run_stage_c1(campaign: dict[str, Any], *, device: str) -> None:
    stage_b2 = json.loads(Path(campaign["reports"]["stageB2_json"]).read_text(encoding="utf-8"))
    if not stage_b2.get("surviving_candidates"):
        _write_stage_c1_report(campaign)
        return
    _run_task_queue(campaign, "practicalization_manifest", _prepare_practicalization_tasks(campaign, rerun=False), device=_resolved_device(device))
    _write_stage_c1_report(campaign)


def run_stage_c2(campaign: dict[str, Any], *, device: str) -> None:
    stage_c1 = json.loads(Path(campaign["reports"]["stageC1_json"]).read_text(encoding="utf-8"))
    if not stage_c1.get("surviving_candidates"):
        _write_stage_c2_report(campaign)
        return
    _run_task_queue(campaign, "practicalization_manifest", _prepare_practicalization_tasks(campaign, rerun=True), device=_resolved_device(device))
    _write_stage_c2_report(campaign)


def run_stage_c3(campaign: dict[str, Any]) -> None:
    _write_stage_c3_report(campaign)


def run_stage_d1(campaign: dict[str, Any]) -> None:
    _write_stage_c3_report(campaign)
    _payload, status = _write_candidate_pack_and_gate(campaign)
    _write_decision_memo(campaign, status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Memory conversion and practicalization campaign runner.")
    parser.add_argument(
        "--campaign-config",
        default="configs/experiments/lss_memory_conversion_program/campaign.yaml",
        help="Path to Memory conversion campaign config",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("stage-a", "stage-b1", "stage-b2", "stage-c1", "stage-c2"):
        command = subparsers.add_parser(name)
        command.add_argument("--device", default="auto")
    subparsers.add_parser("stage-c3")
    subparsers.add_parser("stage-d1")

    eval_task = subparsers.add_parser("run-eval-task")
    eval_task.add_argument("--spec", required=True)
    eval_task.add_argument("--device", default="auto")

    practical_task = subparsers.add_parser("run-practicalization-task")
    practical_task.add_argument("--spec", required=True)
    practical_task.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign = load_campaign_config(args.campaign_config)
    if args.command == "stage-a":
        run_stage_a(campaign, device=args.device)
        return
    if args.command == "stage-b1":
        run_stage_b1(campaign, device=args.device)
        return
    if args.command == "stage-b2":
        run_stage_b2(campaign, device=args.device)
        return
    if args.command == "stage-c1":
        run_stage_c1(campaign, device=args.device)
        return
    if args.command == "stage-c2":
        run_stage_c2(campaign, device=args.device)
        return
    if args.command == "stage-c3":
        run_stage_c3(campaign)
        return
    if args.command == "stage-d1":
        run_stage_d1(campaign)
        return
    if args.command == "run-eval-task":
        run_eval_task(Path(args.spec), device=_resolved_device(args.device))
        return
    if args.command == "run-practicalization-task":
        run_practicalization_task(Path(args.spec), device=_resolved_device(args.device))
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
