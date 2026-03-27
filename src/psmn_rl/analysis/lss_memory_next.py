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
        "# Memory Next State Reconciliation",
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
        "- The first real greedy-conversion signal on this branch came from actor-hidden FiLM, and the best confirmed local point is `partial_shift22`.",
        "- This program therefore focuses on Memory sampled-to-greedy conversion, practicalization, and task-scoped benchmarking, not a reopened broad architecture league.",
    ]
    _ensure_parent(campaign["reports"]["state_reconciliation"]).write_text("\n".join(state_lines) + "\n", encoding="utf-8")

    baseline_lines = [
        "# Memory Next Baseline Sync",
        "",
        "| Surface | Current Accepted Summary | Reference |",
        "| --- | --- | --- |",
        "| `round6` | active DoorKey benchmark retained | `outputs/reports/next_arch_wave_decision_memo.md` |",
        "| `Memory` recurrent fairness | fair recurrent control lane repaired but weak | `outputs/reports/recurrent_fairness_stage1_controls.md` |",
        "| `Memory` POR nearby branch | sampled/temperature-sensitive only | `outputs/reports/por2_nearby_followup.md` |",
        "| `Memory` sampled headroom | strongest bounded result is `POR switchy` `seed7` | `outputs/reports/next_arch_wave_memory_sampled_eval_followup.md` |",
        "| `Memory` near-deterministic boundary | strongest gap sits around `0.05` | `outputs/reports/next_arch_wave_memory_por_boundary.md` |",
        "| `Memory` recurrent comparison | strongest control shares part of the effect but stays below `POR` | `outputs/reports/next_arch_wave_memory_control_variant_thresholds.md` |",
        "| local greedy-conversion point | `partial_shift22` is the best confirmed actor-hidden line | `outputs/reports/memory_actor_hidden_threshold_confirmation.md` |",
        "",
        "## Interpretation",
        "",
        "- The repo state is internally consistent enough to proceed.",
        "- The Memory branch is real enough to justify a focused conversion program, but not real enough yet to justify any benchmark promotion claim.",
    ]
    _ensure_parent(campaign["reports"]["baseline_sync"]).write_text("\n".join(baseline_lines) + "\n", encoding="utf-8")

    registration_lines = [
        "# Memory Next Registration",
        "",
        f"- target substantive runs: `{campaign['target_substantive_runs']}`",
        "- 50/50 program split:",
        "  - exploit track: `40` substantive candidates",
        "  - exploration track: `40` substantive candidates",
        "",
        "## Memory Groups",
        "",
    ]
    for key in ("dev_groups", "holdout_groups", "healthy_groups", "stability_groups"):
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
            "## Families",
            "",
            "- exploit:",
        ]
    )
    for family_key, family in campaign["conversion_families"].items():
        registration_lines.append(
            f"  - `{family_key}`: `{family['label']}` with `{len(family['variants'])}` variants"
        )
    registration_lines.extend(
        [
            "- explore:",
        ]
    )
    for family_key, family in campaign["practicalization"]["families"].items():
        registration_lines.append(
            f"  - `{family_key}`: `{family['label']}`"
        )
    registration_lines.extend(
        [
            "",
            "## Decision Rules",
            "",
            "- Stage B and Stage C both use calibration plus fresh-root rerun before a family is judged alive or dead.",
            "- Stage D advances only candidates that rerun directionally, hold on Memory holdout, and avoid obvious healthy/stability regressions.",
            "- Nothing changes in the accepted benchmark state unless a Memory candidate survives fairness, holdout, stability, and the final candidate-pack / gate path.",
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
        "psmn_rl.analysis.lss_memory_next",
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
    for checkpoint_key in ("por_nearby_switchy_7", "gru_long_11", "partial_shift22", "por_nearby_base_11"):
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


def _top_rows_by_family(rows: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family_key"])].append(row)
    selected: list[dict[str, Any]] = []
    for family_key in sorted(grouped):
        subset = sorted(
            grouped[family_key],
            key=lambda row: (
                float(row.get("candidate_success_rate", row.get("after_greedy_success", 0.0))),
                float(row.get("greedy_success_rate", row.get("after_best_sampled_success", 0.0))),
                -float(row.get("complete_group_failures", 0.0)),
            ),
            reverse=True,
        )
        selected.extend(subset[:top_k])
    return selected


def _prepare_exploit_tasks(campaign: dict[str, Any], *, phase: str) -> list[dict[str, Any]]:
    stage_root_map = {
        "screening": Path(campaign["stage_roots"]["exploit_screening"]),
        "rerun": Path(campaign["stage_roots"]["exploit_rerun"]),
        "verification": Path(campaign["stage_roots"]["exploit_verification"]),
        "holdout": Path(campaign["stage_roots"]["exploit_holdout"]) / "exploit",
        "healthy": Path(campaign["stage_roots"]["exploit_healthy"]) / "healthy",
        "stability": Path(campaign["stage_roots"]["exploit_healthy"]) / "stability",
    }
    groups_map = {
        "screening": list(campaign["analysis"]["dev_groups"]),
        "rerun": list(campaign["analysis"]["dev_groups"]),
        "verification": list(campaign["analysis"]["dev_groups"]),
        "holdout": list(campaign["analysis"]["holdout_groups"]),
        "healthy": list(campaign["analysis"]["healthy_groups"]),
        "stability": list(campaign["analysis"]["stability_groups"]),
    }
    stage_root = stage_root_map[phase]
    groups = groups_map[phase]
    selected: set[str] | None = None
    if phase == "rerun":
        calibration_rows = [_candidate_summary_row(row) for row in _collect_task_summaries(campaign["stage_roots"]["exploit_screening"])]
        selected = {str(row["candidate_id"]) for row in _top_rows_by_family(calibration_rows, top_k=2)}
    elif phase in {"verification", "holdout", "healthy", "stability"}:
        if not Path(campaign["reports"]["stageB1_json"]).exists():
            return []
        payload = json.loads(Path(campaign["reports"]["stageB1_json"]).read_text(encoding="utf-8"))
        selected = {str(row["candidate_id"]) for row in payload.get("surviving_candidates", [])}
        if not selected:
            return []
    tasks: list[dict[str, Any]] = []
    for family_key, family_label, candidate in _expand_conversion_variants(campaign):
        candidate_id = str(candidate["label"])
        if selected is not None and candidate_id not in selected:
            continue
        checkpoint = _checkpoint_spec(campaign, str(candidate["checkpoint"]))
        output_root = stage_root / family_key / candidate_id
        spec = {
            "kind": f"exploit_{phase}",
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
        if phase in {"verification", "holdout", "healthy", "stability"}:
            spec["anchor_modes"] = [
                {"label": "lower_anchor_t05", "strategy": "constant_temp", "temperature": 0.05, "checkpoint_key": checkpoint["key"]},
                {"label": "gap_anchor_t055", "strategy": "constant_temp", "temperature": 0.055, "checkpoint_key": checkpoint["key"]},
                {"label": "shoulder_anchor_t08", "strategy": "constant_temp", "temperature": 0.08, "checkpoint_key": checkpoint["key"]},
            ]
        spec_path = output_root / "task_spec.json"
        _write_task_spec(spec_path, spec)
        tasks.append(
            {
                "job_id": f"exploit_{phase}__{family_key}__{candidate_id}",
                "runner": "run-eval-task",
                "family_label": family_label,
                "task_label": "Memory",
                "variant": candidate_id,
                "seed_block": f"{phase}_groups",
                "config_path": checkpoint["config_path"],
                "config_hash": _sha256_path(spec_path),
                "output_root": str(output_root),
                "rerun_lineage": "fresh_rerun" if phase in {"rerun", "verification", "holdout", "healthy", "stability"} else "original",
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


def _write_band_definition_report(campaign: dict[str, Any]) -> None:
    bands = campaign["analysis"]["temperature_bands"]
    lines = [
        "# Memory Next Band Definition",
        "",
        f"- current best local greedy-conversion point: `{campaign['analysis']['current_best_local_point']}`",
        "",
        "## Temperature Bands",
        "",
        f"- lower boundary band: `{bands['lower_boundary']}`",
        f"- strongest-gap band: `{bands['strongest_gap']}`",
        f"- shoulder / fade-out band: `{bands['shoulder']}`",
        f"- upper plateau reference: `{bands['upper_plateau']}`",
        "",
        "## Evaluation Splits",
        "",
    ]
    for key in ("dev_groups", "holdout_groups", "healthy_groups", "stability_groups"):
        lines.append(f"- `{key}`:")
        for group in campaign["analysis"][key]:
            lines.append(
                f"  - `{group['label']}`: reset seeds `{int(group['start_seed'])}`–`{int(group['start_seed']) + int(group['episodes']) - 1}`"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The Memory branch is now registered as a real benchmark family with explicit lower, gap, and shoulder bands rather than one attractive curve.",
            "- `partial_shift22` is the actor-hidden local greedy-conversion incumbent; `POR switchy seed7` and `gru_long_11` remain the non-practicalized baselines for fair comparison.",
        ]
    )
    _ensure_parent(campaign["reports"]["band_definition"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    partial22_greedy = 0.0
    partial22_gap = 0.0
    for checkpoint_key, checkpoint_rows in grouped.items():
        greedy = next((row for row in checkpoint_rows if str(row["candidate"]["label"]) == "greedy"), None)
        t005 = next((row for row in checkpoint_rows if str(row["candidate"]["label"]) == "lower_t005"), None)
        t055 = next((row for row in checkpoint_rows if str(row["candidate"]["label"]) == "gap_t055"), None)
        if greedy and t005:
            best_gap = max(best_gap, float(t005["candidate_overall"]["success_rate"]) - float(greedy["candidate_overall"]["success_rate"]))
        if greedy and t055:
            best_gap = max(best_gap, float(t055["candidate_overall"]["success_rate"]) - float(greedy["candidate_overall"]["success_rate"]))
        if checkpoint_key == "por_nearby_switchy_7" and t005:
            por_edge = float(t005["candidate_overall"]["success_rate"])
        if checkpoint_key == "gru_long_11" and t005:
            por_edge -= float(t005["candidate_overall"]["success_rate"])
        if checkpoint_key == "partial_shift22":
            partial22_greedy = float(greedy["candidate_overall"]["success_rate"]) if greedy else 0.0
            partial22_gap = float(t055["candidate_overall"]["success_rate"]) if t055 else 0.0
    best_probe_lift = max((float(row.get("lift_vs_majority", 0.0)) for row in probe_results if row.get("status") == "completed"), default=0.0)
    if partial22_greedy >= 0.20 and best_gap >= 0.20:
        diagnosis = "both, with decode/selection primary"
    elif best_gap >= 0.20 and best_probe_lift >= 0.10:
        diagnosis = "decode/selection failure with exposed persistence signal"
    elif best_gap >= 0.20:
        diagnosis = "decode/selection failure"
    else:
        diagnosis = "state/source-quality ceiling"
    lines = [
        "# Memory Next Source Quality",
        "",
        f"- diagnosis: `{diagnosis}`",
        f"- strongest sampled-to-greedy gap: `{_format_float(best_gap)}`",
        f"- POR lower-band edge over strongest recurrent control: `{_format_float(por_edge)}`",
        f"- actor-hidden local greedy point (`partial_shift22`) : greedy `{_format_float(partial22_greedy)}`, gap-band `{_format_float(partial22_gap)}`",
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
            "- The probe fits show that persistence-related state is already exposed enough to decode shallow targets from frozen activations, but exact greedy still fails on the non-practicalized lines.",
            "- `partial_shift22` demonstrates that the branch is not an absolute state ceiling, but the remaining gap still looks dominated by decode/selection and objective-shaping limits.",
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
        "complete_group_failures": float(sum(1 for group in summary["candidate_groups"] if float(group["success_rate"]) <= 0.0)),
        "rerun_lineage": str(summary["rerun_lineage"]),
    }


def _baseline_success(rows: list[dict[str, Any]], candidate_id: str) -> float:
    match = next((row for row in rows if str(row["candidate_id"]) == candidate_id), None)
    return float(match["candidate_success_rate"]) if match else 0.0


def _best_baseline(rows: list[dict[str, Any]], candidate_ids: list[str]) -> float:
    return max((_baseline_success(rows, candidate_id) for candidate_id in candidate_ids), default=0.0)


def _write_stage_b1_report(campaign: dict[str, Any]) -> dict[str, Any]:
    calibration_rows = [_candidate_summary_row(row) for row in _collect_task_summaries(campaign["stage_roots"]["exploit_screening"])]
    rerun_rows = {_candidate_summary_row(row)["candidate_id"]: _candidate_summary_row(row) for row in _collect_task_summaries(campaign["stage_roots"]["exploit_rerun"])}
    por_baseline = _baseline_success(calibration_rows, "select_por_switchy7_t055")
    recurrent_baseline = _baseline_success(calibration_rows, "select_gru_long11_t055")
    actor_hidden_baseline = _best_baseline(calibration_rows, ["select_partial22_greedy", "partial22_greedy"])
    selected_candidates: list[dict[str, Any]] = []
    surviving_candidates: list[dict[str, Any]] = []
    for row in _top_rows_by_family(calibration_rows, top_k=2):
        rerun = rerun_rows.get(str(row["candidate_id"]))
        selected = {
            **row,
            "rerun_success_rate": float(rerun["candidate_success_rate"]) if rerun is not None else None,
            "rerun_greedy_success_rate": float(rerun["greedy_success_rate"]) if rerun is not None else None,
        }
        selected_candidates.append(selected)
        if rerun is not None:
            stable = float(rerun["candidate_success_rate"]) >= float(row["candidate_success_rate"]) - 0.05
            edge = (
                float(rerun["candidate_success_rate"]) >= max(por_baseline, recurrent_baseline, actor_hidden_baseline) - 0.01
                or float(rerun["greedy_success_rate"]) >= actor_hidden_baseline - 0.01
            )
            if stable and edge:
                surviving_candidates.append(selected)
    lines = [
        "# Memory Next Stage B1 Exploit Screening",
        "",
        f"- substantive exploit candidates: `{len(calibration_rows)}`",
        f"- family rerun selections: `{len(selected_candidates)}`",
        f"- Stage B1 survivors: `{len(surviving_candidates)}`",
        f"- POR dev incumbent (`select_por_switchy7_t055`): `{_format_float(por_baseline)}`",
        f"- recurrent dev incumbent (`select_gru_long11_t055`): `{_format_float(recurrent_baseline)}`",
        f"- actor-hidden dev incumbent (`partial_shift22` greedy): `{_format_float(actor_hidden_baseline)}`",
        "",
    ]
    table_rows: list[list[str]] = []
    for row in sorted(calibration_rows, key=lambda item: (str(item["family_key"]), -float(item["candidate_success_rate"]))):
        rerun = rerun_rows.get(str(row["candidate_id"]))
        table_rows.append(
            [
                str(row["family_label"]),
                str(row["candidate_id"]),
                str(row["checkpoint_variant"]),
                _format_float(row["greedy_success_rate"]),
                _format_float(row["candidate_success_rate"]),
                _format_float(float(rerun["candidate_success_rate"]) if rerun else None),
                _format_float(float(rerun["greedy_success_rate"]) if rerun else None),
                _format_float(row["candidate_success_rate"] - por_baseline),
                _format_float(row["candidate_success_rate"] - recurrent_baseline),
                _format_float(row["candidate_success_rate"] - actor_hidden_baseline),
                _format_float(row["compute_cost_ratio"], 3),
                str("advance" if any(str(item["candidate_id"]) == str(row["candidate_id"]) for item in surviving_candidates) else "pruned"),
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "Family",
                "Candidate",
                "Checkpoint",
                "Greedy",
                "Dev Candidate",
                "Rerun Candidate",
                "Rerun Greedy",
                "Delta vs POR",
                "Delta vs GRU",
                "Delta vs Actor-Hidden",
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
            "- Each exploit family received a calibration sweep and a fresh-root rerun on its top dev candidates.",
            "- Candidates advance only if they rerun directionally and stay competitive with the current POR, recurrent, or actor-hidden incumbents instead of winning by one noisy dev point.",
        ]
    )
    payload = {
        "calibration_rows": calibration_rows,
        "selected_candidates": selected_candidates,
        "surviving_candidates": surviving_candidates,
        "por_baseline": por_baseline,
        "recurrent_baseline": recurrent_baseline,
        "actor_hidden_baseline": actor_hidden_baseline,
    }
    _ensure_parent(campaign["reports"]["stageB1_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(campaign["reports"]["stageB1_json"], payload)
    return payload


def _collect_practicalization_rows(stage_root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = Path(stage_root)
    if not root.exists():
        return rows
    for summary_path in sorted(root.glob("**/summary.json")):
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        task_spec_path = summary_path.parent / "task_spec.json"
        task_spec = json.loads(task_spec_path.read_text(encoding="utf-8")) if task_spec_path.exists() else {}
        row["run_name"] = summary_path.parent.name
        row["student_label"] = str(task_spec.get("student_label", row.get("variant", summary_path.parent.name)))
        row["family_key"] = str(task_spec.get("family_key", "unknown"))
        row["family_label"] = str(task_spec.get("family_label", "unknown"))
        row["checkpoint_key"] = str(task_spec.get("checkpoint", {}).get("key", "unknown"))
        row["output_root"] = str(summary_path.parent)
        rows.append(row)
    return rows


def _select_practicalization_reruns(rows: list[dict[str, Any]]) -> set[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family_key"])].append(row)
    selected: set[str] = set()
    for family_key, subset in grouped.items():
        ranked = sorted(
            subset,
            key=lambda row: (
                float(row["after_greedy_success"]),
                float(row["after_best_sampled_success"] - row["before_best_sampled_success"]),
                -float(row["harvest_successes"]),
            ),
            reverse=True,
        )[:2]
        for row in ranked:
            selected.add(str(row["run_name"]))
    return selected


def _prepare_practicalization_tasks(campaign: dict[str, Any], *, rerun: bool = False) -> list[dict[str, Any]]:
    stage_root = Path(campaign["stage_roots"]["practicalization_rerun" if rerun else "practicalization_screening"])
    tasks: list[dict[str, Any]] = []
    rerun_filter: set[str] | None = None
    if rerun:
        screening_rows = _collect_practicalization_rows(campaign["stage_roots"]["practicalization_screening"])
        rerun_filter = _select_practicalization_reruns(screening_rows)
    student_map = {str(student["student_label"]): student for student in campaign["practicalization"]["students"]}
    for student_label, student in student_map.items():
        checkpoint = _checkpoint_spec(campaign, str(student["checkpoint"]))
        for family_key, family in campaign["practicalization"]["families"].items():
            allowed_students = {str(label) for label in family.get("student_keys", list(student_map.keys()))}
            if student_label not in allowed_students:
                continue
            for temperature in family["temperatures"]:
                for target in family["targets"]:
                    for weighting in family["weightings"]:
                        run_name = f"{student_label}__{family_key}__t{str(temperature).replace('.', 'p')}__{target}__{weighting}"
                        if rerun_filter is not None and run_name not in rerun_filter:
                            continue
                        output_root = stage_root / run_name
                        spec = {
                            "checkpoint": checkpoint,
                            "student_label": student_label,
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
                            "groups": list(campaign["analysis"]["dev_groups"]),
                            "max_steps": int(campaign["analysis"]["max_steps"]),
                            "output_root": str(output_root),
                        }
                        spec_path = output_root / "task_spec.json"
                        _write_task_spec(spec_path, spec)
                        tasks.append(
                            {
                                "job_id": f"{'explore_rerun' if rerun else 'explore'}__{run_name}",
                                "runner": "run-practicalization-task",
                                "family_label": str(family["label"]),
                                "task_label": "Memory",
                                "variant": run_name,
                                "seed_block": student_label,
                                "config_path": checkpoint["config_path"],
                                "config_hash": _sha256_path(spec_path),
                                "output_root": str(output_root),
                                "rerun_lineage": "fresh_rerun" if rerun else "original",
                                "spec_path": spec_path,
                            }
                        )
    return tasks


def _write_stage_c1_report(campaign: dict[str, Any]) -> dict[str, Any]:
    rows = _collect_practicalization_rows(campaign["stage_roots"]["practicalization_screening"])
    rerun_rows = {str(row["run_name"]): row for row in _collect_practicalization_rows(campaign["stage_roots"]["practicalization_rerun"])}
    if not rows:
        _write_skip_report(campaign["reports"]["stageC1_report"], "Memory Next Stage C1 Explore Screening", "no exploration runs completed")
        payload = {"rows": [], "selected_candidates": [], "surviving_candidates": []}
        _write_json(campaign["reports"]["stageC1_json"], payload)
        return payload
    stage_b1_path = Path(campaign["reports"]["stageB1_json"])
    actor_hidden_baseline = 0.0
    if stage_b1_path.exists():
        actor_hidden_baseline = float(json.loads(stage_b1_path.read_text(encoding="utf-8")).get("actor_hidden_baseline", 0.0))
    selected_names = _select_practicalization_reruns(rows)
    selected_candidates: list[dict[str, Any]] = []
    surviving_candidates: list[dict[str, Any]] = []
    for row in rows:
        if str(row["run_name"]) not in selected_names:
            continue
        rerun = rerun_rows.get(str(row["run_name"]))
        selected = {
            "run_name": str(row["run_name"]),
            "family_key": str(row["family_key"]),
            "family_label": str(row["family_label"]),
            "student_label": str(row["student_label"]),
            "teacher_temperature": float(row["teacher_temperature"]),
            "before_greedy_success": float(row["before_greedy_success"]),
            "after_greedy_success": float(row["after_greedy_success"]),
            "before_best_sampled_success": float(row["before_best_sampled_success"]),
            "after_best_sampled_success": float(row["after_best_sampled_success"]),
            "rerun_after_greedy_success": float(rerun["after_greedy_success"]) if rerun else None,
            "rerun_after_best_sampled_success": float(rerun["after_best_sampled_success"]) if rerun else None,
        }
        selected_candidates.append(selected)
        if rerun is not None:
            stable = (
                float(rerun["after_greedy_success"]) >= float(row["after_greedy_success"]) - 0.05
                and float(rerun["after_best_sampled_success"]) >= float(row["after_best_sampled_success"]) - 0.05
            )
            greedy_edge = float(rerun["after_greedy_success"]) >= actor_hidden_baseline + 0.01
            sampled_edge = (
                float(rerun["after_best_sampled_success"]) >= float(row["before_best_sampled_success"]) + 0.05
                and float(rerun["after_greedy_success"]) >= float(row["before_greedy_success"]) - 0.02
            )
            if stable and (greedy_edge or sampled_edge):
                surviving_candidates.append(selected)
    lines = [
        "# Memory Next Stage C1 Explore Screening",
        "",
        f"- substantive exploration candidates: `{len(rows)}`",
        f"- family rerun selections: `{len(selected_candidates)}`",
        f"- Stage C1 survivors: `{len(surviving_candidates)}`",
        f"- actor-hidden greedy incumbent (`partial_shift22`): `{_format_float(actor_hidden_baseline)}`",
        "",
    ]
    lines.extend(
        _markdown_table(
            [
                "Family",
                "Run",
                "Student",
                "Teacher Temp",
                "Before Greedy",
                "After Greedy",
                "After Best Sampled",
                "Rerun Greedy",
                "Rerun Best Sampled",
                "Decision",
            ],
            [
                [
                    str(row["family_label"]),
                    str(row["run_name"]),
                    str(row["student_label"]),
                    _format_float(row["teacher_temperature"], 3),
                    _format_float(row["before_greedy_success"]),
                    _format_float(row["after_greedy_success"]),
                    _format_float(row["after_best_sampled_success"]),
                    _format_float(rerun_rows[str(row["run_name"])]["after_greedy_success"]) if str(row["run_name"]) in rerun_rows else "-",
                    _format_float(rerun_rows[str(row["run_name"])]["after_best_sampled_success"]) if str(row["run_name"]) in rerun_rows else "-",
                    str("advance" if any(str(item["run_name"]) == str(row["run_name"]) for item in surviving_candidates) else "pruned"),
                ]
                for row in sorted(rows, key=lambda item: (str(item["family_key"]), -float(item["after_greedy_success"])))
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The exploration track is judged on the actual low-temperature Memory bands rather than generic high-temperature policy diagnostics.",
            "- Candidates advance only if the fresh-root rerun preserves the direction and moves either greedy conversion or sampled-band quality enough to matter against the current actor-hidden incumbent.",
        ]
    )
    payload = {"rows": rows, "selected_candidates": selected_candidates, "surviving_candidates": surviving_candidates}
    _ensure_parent(campaign["reports"]["stageC1_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(campaign["reports"]["stageC1_json"], payload)
    return payload


def _write_stage_d1_report(campaign: dict[str, Any]) -> dict[str, Any]:
    stage_b1 = _load_stage_b1_payload(campaign) if Path(campaign["reports"]["stageB1_json"]).exists() else {"surviving_candidates": []}
    stage_c1 = json.loads(Path(campaign["reports"]["stageC1_json"]).read_text(encoding="utf-8")) if Path(campaign["reports"]["stageC1_json"]).exists() else {"surviving_candidates": []}
    exploit_rows = [
        {
            "candidate_id": str(row["candidate_id"]),
            "family_key": str(row["family_key"]),
            "screen_success": float(row["candidate_success_rate"]),
            "rerun_success": float(row["rerun_success_rate"]),
            "rerun_greedy_success": float(row["rerun_greedy_success_rate"]),
        }
        for row in stage_b1.get("surviving_candidates", [])
    ]
    explore_rows = [
        {
            "run_name": str(row["run_name"]),
            "family_key": str(row["family_key"]),
            "rerun_after_greedy_success": float(row["rerun_after_greedy_success"]),
            "rerun_after_best_sampled_success": float(row["rerun_after_best_sampled_success"]),
        }
        for row in stage_c1.get("surviving_candidates", [])
    ]
    lines = [
        "# Memory Next Stage D1 Verification",
        "",
        f"- exploit survivors entering verification: `{len(exploit_rows)}`",
        f"- exploration survivors entering verification: `{len(explore_rows)}`",
        "",
    ]
    if exploit_rows:
        lines.extend(
            [
                "## Exploit Rerun Consolidation",
                "",
            ]
        )
        lines.extend(
            _markdown_table(
                ["Candidate", "Family", "Screen Success", "Rerun Success", "Rerun Greedy"],
                [
                    [
                        str(row["candidate_id"]),
                        str(row["family_key"]),
                        _format_float(row["screen_success"]),
                        _format_float(row["rerun_success"]),
                        _format_float(row["rerun_greedy_success"]),
                    ]
                    for row in exploit_rows
                ],
            )
        )
    if explore_rows:
        lines.extend(
            [
                "",
                "## Explore Rerun Consolidation",
                "",
            ]
        )
        lines.extend(
            _markdown_table(
                ["Run", "Family", "Rerun Greedy", "Rerun Best Sampled"],
                [
                    [
                        str(row["run_name"]),
                        str(row["family_key"]),
                        _format_float(row["rerun_after_greedy_success"]),
                        _format_float(row["rerun_after_best_sampled_success"]),
                    ]
                    for row in explore_rows
                ],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage D1 consolidates the fresh-root reruns that were already required by the family fair-shot rule.",
            "- Only directionally consistent rerun lines are allowed to enter holdout and anti-regression work.",
        ]
    )
    payload = {"exploit_rows": exploit_rows, "explore_rows": explore_rows}
    _ensure_parent(campaign["reports"]["stageD1_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(campaign["reports"]["stageD1_json"], payload)
    return payload


def _practicalized_checkpoint_spec(run_dir: Path) -> dict[str, Any]:
    task_spec = json.loads((run_dir / "task_spec.json").read_text(encoding="utf-8"))
    return {
        "key": run_dir.name,
        "family": "practicalized",
        "family_label": str(task_spec["family_label"]),
        "variant": str(task_spec["student_label"]),
        "seed": 0,
        "config_path": str(run_dir / "resolved_config.yaml"),
        "checkpoint_path": str(run_dir / "latest.pt"),
    }


def _prepare_explore_eval_tasks(campaign: dict[str, Any], *, phase: str) -> list[dict[str, Any]]:
    if not Path(campaign["reports"]["stageC1_json"]).exists():
        return []
    stage_c1 = json.loads(Path(campaign["reports"]["stageC1_json"]).read_text(encoding="utf-8"))
    survivors = stage_c1.get("surviving_candidates", [])
    if not survivors:
        return []
    group_key = {"holdout": "holdout_groups", "healthy": "healthy_groups", "stability": "stability_groups"}[phase]
    root_base = {
        "holdout": Path(campaign["stage_roots"]["exploit_holdout"]) / "explore",
        "healthy": Path(campaign["stage_roots"]["exploit_healthy"]) / "explore_healthy",
        "stability": Path(campaign["stage_roots"]["exploit_healthy"]) / "explore_stability",
    }[phase]
    tasks: list[dict[str, Any]] = []
    for survivor in survivors:
        run_name = str(survivor["run_name"])
        run_dir = Path(campaign["stage_roots"]["practicalization_rerun"]) / run_name
        if not (run_dir / "latest.pt").exists():
            continue
        checkpoint = _practicalized_checkpoint_spec(run_dir)
        output_root = root_base / run_name
        task_spec_path = output_root / "task_spec.json"
        task_spec = json.loads((run_dir / "task_spec.json").read_text(encoding="utf-8"))
        candidate = {
            "label": "teacher_temp",
            "strategy": "constant_temp",
            "temperature": float(task_spec["temperature"]),
            "checkpoint_key": checkpoint["key"],
            "family_key": str(survivor["family_key"]),
            "family_label": str(survivor["family_label"]),
        }
        spec = {
            "kind": f"explore_{phase}",
            "checkpoint_key": checkpoint["key"],
            "checkpoint": checkpoint,
            "candidate": candidate,
            "groups": list(campaign["analysis"][group_key]),
            "max_steps": int(campaign["analysis"]["max_steps"]),
            "collect_probes": False,
            "probe_episode_limit": 0,
            "probe_draws": 0,
            "anchor_modes": [
                {"label": "lower_anchor_t05", "strategy": "constant_temp", "temperature": 0.05, "checkpoint_key": checkpoint["key"]},
                {"label": "gap_anchor_t055", "strategy": "constant_temp", "temperature": 0.055, "checkpoint_key": checkpoint["key"]},
                {"label": "shoulder_anchor_t08", "strategy": "constant_temp", "temperature": 0.08, "checkpoint_key": checkpoint["key"]},
            ],
            "output_root": str(output_root),
        }
        _write_task_spec(task_spec_path, spec)
        tasks.append(
            {
                "job_id": f"explore_{phase}__{run_name}",
                "runner": "run-eval-task",
                "family_label": str(survivor["family_label"]),
                "task_label": "Memory",
                "variant": run_name,
                "seed_block": group_key,
                "config_path": checkpoint["config_path"],
                "config_hash": _sha256_path(task_spec_path),
                "output_root": str(output_root),
                "rerun_lineage": "fresh_rerun",
                "spec_path": task_spec_path,
            }
        )
    return tasks


def _prepare_holdout_control_tasks(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    holdout_controls = {"select_por_switchy7_t055", "select_gru_long11_t055", "select_partial22_greedy"}
    tasks = _prepare_exploit_tasks(campaign, phase="holdout")
    extra: list[dict[str, Any]] = []
    seen = {task["variant"] for task in tasks}
    for family_key, family_label, candidate in _expand_conversion_variants(campaign):
        candidate_id = str(candidate["label"])
        if candidate_id not in holdout_controls or candidate_id in seen:
            continue
        checkpoint = _checkpoint_spec(campaign, str(candidate["checkpoint"]))
        output_root = Path(campaign["stage_roots"]["exploit_holdout"]) / "controls" / family_key / candidate_id
        spec = {
            "kind": "exploit_holdout_control",
            "checkpoint_key": checkpoint["key"],
            "checkpoint": checkpoint,
            "candidate": {**candidate, "checkpoint_key": checkpoint["key"]},
            "groups": list(campaign["analysis"]["holdout_groups"]),
            "max_steps": int(campaign["analysis"]["max_steps"]),
            "collect_probes": False,
            "probe_episode_limit": 0,
            "probe_draws": 0,
            "anchor_modes": [
                {"label": "lower_anchor_t05", "strategy": "constant_temp", "temperature": 0.05, "checkpoint_key": checkpoint["key"]},
                {"label": "gap_anchor_t055", "strategy": "constant_temp", "temperature": 0.055, "checkpoint_key": checkpoint["key"]},
                {"label": "shoulder_anchor_t08", "strategy": "constant_temp", "temperature": 0.08, "checkpoint_key": checkpoint["key"]},
            ],
            "output_root": str(output_root),
        }
        spec_path = output_root / "task_spec.json"
        _write_task_spec(spec_path, spec)
        extra.append(
            {
                "job_id": f"exploit_holdout_control__{family_key}__{candidate_id}",
                "runner": "run-eval-task",
                "family_label": family_label,
                "task_label": "Memory",
                "variant": candidate_id,
                "seed_block": "holdout_groups",
                "config_path": checkpoint["config_path"],
                "config_hash": _sha256_path(spec_path),
                "output_root": str(output_root),
                "rerun_lineage": "fresh_rerun",
                "spec_path": spec_path,
            }
        )
    return tasks + extra


def _write_stage_d2_report(campaign: dict[str, Any]) -> dict[str, Any]:
    rows = _collect_task_summaries(campaign["stage_roots"]["exploit_holdout"])
    if not rows:
        _write_skip_report(campaign["reports"]["stageD2_report"], "Memory Next Stage D2 Fairness Holdout", "no candidate survived to holdout")
        payload = {"rows": [], "surviving_candidates": []}
        _write_json(campaign["reports"]["stageD2_json"], payload)
        return payload
    control_rows = {str(row["candidate"]["label"]): row for row in rows if str(row["candidate"]["label"]) in {"select_por_switchy7_t055", "select_gru_long11_t055", "select_partial22_greedy"}}
    por_gap = float(control_rows.get("select_por_switchy7_t055", {}).get("anchors", {}).get("gap_anchor_t055", {}).get("overall", {}).get("success_rate", 0.0))
    gru_gap = float(control_rows.get("select_gru_long11_t055", {}).get("anchors", {}).get("gap_anchor_t055", {}).get("overall", {}).get("success_rate", 0.0))
    arch_greedy = float(control_rows.get("select_partial22_greedy", {}).get("greedy_overall", {}).get("success_rate", 0.0))
    surviving_candidates: list[dict[str, Any]] = []
    table_rows: list[list[str]] = []
    for row in rows:
        candidate_id = str(row["candidate"]["label"])
        lower = float(row["anchors"].get("lower_anchor_t05", {}).get("overall", {}).get("success_rate", 0.0))
        gap = float(row["anchors"].get("gap_anchor_t055", {}).get("overall", {}).get("success_rate", 0.0))
        shoulder = float(row["anchors"].get("shoulder_anchor_t08", {}).get("overall", {}).get("success_rate", 0.0))
        greedy = float(row["greedy_overall"]["success_rate"])
        candidate_success = float(row["candidate_overall"]["success_rate"])
        decision = "control"
        if candidate_id not in {"select_por_switchy7_t055", "select_gru_long11_t055", "select_partial22_greedy"}:
            if gap >= max(por_gap, gru_gap) - 0.02 and greedy >= arch_greedy - 0.05:
                surviving_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "family_key": str(row["candidate"]["family_key"]),
                        "greedy_success": greedy,
                        "lower_success": lower,
                        "gap_success": gap,
                        "shoulder_success": shoulder,
                    }
                )
                decision = "advance"
            else:
                decision = "pruned"
        table_rows.append(
            [
                candidate_id,
                str(row["candidate"]["family_key"]),
                _format_float(greedy),
                _format_float(candidate_success),
                _format_float(lower),
                _format_float(gap),
                _format_float(shoulder),
                decision,
            ]
        )
    lines = [
        "# Memory Next Stage D2 Fairness Holdout",
        "",
        f"- matched holdout runs: `{len(rows)}`",
        f"- stage-d2 survivors: `{len(surviving_candidates)}`",
        f"- holdout POR gap anchor: `{_format_float(por_gap)}`",
        f"- holdout GRU gap anchor: `{_format_float(gru_gap)}`",
        f"- holdout actor-hidden greedy anchor: `{_format_float(arch_greedy)}`",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Candidate", "Family", "Greedy", "Candidate", "Lower", "Gap", "Shoulder", "Decision"],
            table_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage D2 asks whether a candidate still looks real on Memory holdout under matched POR, recurrent-control, and actor-hidden references.",
            "- Surviving candidates must stay meaningful in the strongest-gap band without collapsing on greedy or outside a single narrow point.",
        ]
    )
    payload = {"rows": rows, "surviving_candidates": surviving_candidates}
    _ensure_parent(campaign["reports"]["stageD2_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(campaign["reports"]["stageD2_json"], payload)
    return payload


def _prepare_stage_d3_tasks(campaign: dict[str, Any], *, phase: str) -> list[dict[str, Any]]:
    if not Path(campaign["reports"]["stageD2_json"]).exists():
        return []
    stage_d2 = json.loads(Path(campaign["reports"]["stageD2_json"]).read_text(encoding="utf-8"))
    selected = {str(row["candidate_id"]) for row in stage_d2.get("surviving_candidates", [])}
    if not selected:
        return []
    tasks = _prepare_exploit_tasks(campaign, phase=phase)
    return [task for task in tasks if str(task["variant"]) in selected]


def _write_stage_d3_report(campaign: dict[str, Any]) -> dict[str, Any]:
    healthy_rows = _collect_task_summaries(Path(campaign["stage_roots"]["exploit_healthy"]) / "healthy")
    stability_rows = _collect_task_summaries(Path(campaign["stage_roots"]["exploit_healthy"]) / "stability")
    if not healthy_rows and not stability_rows:
        _write_skip_report(campaign["reports"]["stageD3_report"], "Memory Next Stage D3 Anti-Regression Stability", "no stage-d2 survivor")
        payload = {"surviving_candidates": []}
        _write_json(campaign["reports"]["stageD3_json"], payload)
        return payload
    healthy_map = {str(row["candidate"]["label"]): row for row in healthy_rows}
    stability_map = {str(row["candidate"]["label"]): row for row in stability_rows}
    stage_d2 = json.loads(Path(campaign["reports"]["stageD2_json"]).read_text(encoding="utf-8"))
    surviving_candidates: list[dict[str, Any]] = []
    table_rows: list[list[str]] = []
    for candidate in stage_d2.get("surviving_candidates", []):
        candidate_id = str(candidate["candidate_id"])
        healthy = healthy_map.get(candidate_id)
        stability = stability_map.get(candidate_id)
        healthy_gap = float(healthy["anchors"]["gap_anchor_t055"]["overall"]["success_rate"]) if healthy else 0.0
        stability_greedy = float(stability["greedy_overall"]["success_rate"]) if stability else 0.0
        healthy_duration = float(healthy["candidate_overall"]["option_duration_mean"]) if healthy else 0.0
        healthy_switch = float(healthy["candidate_overall"]["option_switch_mean"]) if healthy else 0.0
        decision = "advance" if healthy_gap >= float(candidate["gap_success"]) - 0.10 and stability_greedy >= float(candidate["greedy_success"]) - 0.10 else "pruned"
        if decision == "advance":
            surviving_candidates.append(
                {
                    **candidate,
                    "healthy_gap_success": healthy_gap,
                    "stability_greedy_success": stability_greedy,
                    "option_duration_mean": healthy_duration,
                    "option_switch_mean": healthy_switch,
                }
            )
        table_rows.append(
            [
                candidate_id,
                _format_float(healthy_gap),
                _format_float(stability_greedy),
                _format_float(healthy_duration),
                _format_float(healthy_switch),
                decision,
            ]
        )
    lines = [
        "# Memory Next Stage D3 Anti-Regression Stability",
        "",
        f"- stage-d3 survivors: `{len(surviving_candidates)}`",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Candidate", "Healthy Gap", "Stability Greedy", "Option Duration", "Option Switch", "Decision"],
            table_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage D3 checks that a candidate is not winning by damaging the rest of the Memory branch.",
            "- Surviving candidates must keep healthy-band behavior and remain stable across the dedicated stability seed groups.",
        ]
    )
    payload = {"surviving_candidates": surviving_candidates}
    _ensure_parent(campaign["reports"]["stageD3_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(campaign["reports"]["stageD3_json"], payload)
    return payload


def _write_candidate_pack_and_gate(campaign: dict[str, Any]) -> tuple[dict[str, Any], str]:
    stage_d3_path = Path(campaign["reports"]["stageD3_json"])
    stage_b1_path = Path(campaign["reports"]["stageB1_json"])
    stage_c1_path = Path(campaign["reports"]["stageC1_json"])
    stage_d3 = json.loads(stage_d3_path.read_text(encoding="utf-8")) if stage_d3_path.exists() else {"surviving_candidates": []}
    stage_b1 = json.loads(stage_b1_path.read_text(encoding="utf-8")) if stage_b1_path.exists() else {"surviving_candidates": []}
    stage_c1 = json.loads(stage_c1_path.read_text(encoding="utf-8")) if stage_c1_path.exists() else {"surviving_candidates": []}
    surviving = list(stage_d3.get("surviving_candidates", []))
    status = "active benchmark confirmed and Memory branch remains exploratory"
    gate_lines = [
        "# Memory Next Gate Report",
        "",
    ]
    candidate_rows: list[dict[str, Any]] = []
    if surviving:
        best = max(surviving, key=lambda row: (float(row["greedy_success"]), float(row["gap_success"])))
        candidate_rows.append(best)
        if float(best["greedy_success"]) >= 0.55 and float(best["gap_success"]) >= 0.50 and stage_c1.get("surviving_candidates"):
            status = "a practicalized Memory candidate becomes benchmark-worthy"
            gate_lines.append("- gate invocation: `task-scoped candidate plus benchmark-lane review`")
            gate_lines.append("- result: `practicalized Memory candidate reached benchmark-worthy evidence inside the Memory lane`")
        else:
            status = "POR or recurrent-control Memory branch earns task-scoped benchmark-candidate status"
            gate_lines.append("- gate invocation: `task-scoped candidate review`")
            gate_lines.append("- result: `Memory candidate survived holdout, controls, and stability strongly enough for task-scoped candidate status`")
    elif stage_b1.get("surviving_candidates") or stage_c1.get("surviving_candidates"):
        status = "active benchmark confirmed and Memory branch remains exploratory"
        gate_lines.append("- gate invocation: `not required`")
        gate_lines.append("- result: `conversion signal remained real, but no candidate survived the full holdout / stability funnel`")
    else:
        status = "the Memory branch narrows further and should not be reopened casually"
        gate_lines.append("- gate invocation: `not required`")
        gate_lines.append("- result: `the branch died before candidate-pack quality evidence existed`")
    payload = {
        "pack_type": "memory_next_summary",
        "generated_at": _timestamp(),
        "active_benchmark": campaign["current_canonical_name"],
        "status": status,
        "memory_candidates": candidate_rows,
    }
    _write_json(campaign["reports"]["candidate_pack"], payload)
    _ensure_parent(campaign["reports"]["gate_report"]).write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
    return payload, status


def _write_decision_memo(campaign: dict[str, Any], status: str) -> None:
    lines = [
        "# Memory Next Decision Memo",
        "",
        f"- final status: `{status}`",
        f"- active benchmark remains: `{campaign['current_canonical_name']}`",
        "",
        "## Summary",
        "",
        "- The large 50/50 Memory program reconciled state, audited source quality, screened both exploit and exploration families, then pushed only surviving candidates through holdout and stability.",
        "- The accepted benchmark state changes only if a Memory candidate survives the same candidate-pack and gate discipline used elsewhere in the repo.",
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
        "rerun_lineage": "fresh_rerun" if "rerun" in str(spec["kind"]) or any(token in str(spec["kind"]) for token in ("verification", "holdout", "healthy", "stability")) else "original",
    }
    _write_json(output_root / "summary.json", summary)


def run_practicalization_task(spec_path: Path, *, device: str) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    checkpoint = dict(spec["checkpoint"])
    output_dir = Path(spec["output_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(checkpoint["config_path"])
    dump_config(config, output_dir / "resolved_config.yaml")
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
    original_checkpoint = torch.load(checkpoint["checkpoint_path"], map_location="cpu", weights_only=False)
    original_checkpoint["model"] = model.state_dict()
    original_checkpoint["memory_next"] = {
        "student_label": spec["student_label"],
        "family_key": spec["family_key"],
        "temperature": spec["temperature"],
        "target": spec["target"],
        "weighting": spec["weighting"],
        **fine_tune_metrics,
    }
    torch.save(original_checkpoint, output_dir / "latest.pt")

    device_t = torch.device("cpu" if _resolved_device(device) == "cpu" or not torch.cuda.is_available() else "cuda")
    mode_specs = {
        "greedy": {"label": "greedy", "strategy": "greedy", "checkpoint_key": checkpoint["key"]},
        "lower_t05": {"label": "lower_t05", "strategy": "constant_temp", "temperature": 0.05, "checkpoint_key": checkpoint["key"]},
        "gap_t055": {"label": "gap_t055", "strategy": "constant_temp", "temperature": 0.055, "checkpoint_key": checkpoint["key"]},
        "shoulder_t08": {"label": "shoulder_t08", "strategy": "constant_temp", "temperature": 0.08, "checkpoint_key": checkpoint["key"]},
    }
    before_modes: dict[str, dict[str, Any]] = {}
    after_modes: dict[str, dict[str, Any]] = {}
    for label, candidate in mode_specs.items():
        before_modes[label] = _evaluate_mode(
            config_path=Path(checkpoint["config_path"]),
            checkpoint_path=Path(checkpoint["checkpoint_path"]),
            candidate=candidate,
            groups=list(spec["groups"]),
            device=device_t,
            max_steps=int(spec["max_steps"]),
            collect_probes=False,
        )
        after_modes[label] = _evaluate_mode(
            config_path=output_dir / "resolved_config.yaml",
            checkpoint_path=output_dir / "latest.pt",
            candidate=candidate,
            groups=list(spec["groups"]),
            device=device_t,
            max_steps=int(spec["max_steps"]),
            collect_probes=False,
        )
    before_sampled = {key: value["overall"]["success_rate"] for key, value in before_modes.items() if key != "greedy"}
    after_sampled = {key: value["overall"]["success_rate"] for key, value in after_modes.items() if key != "greedy"}
    before_best_mode = max(before_sampled, key=before_sampled.get)
    after_best_mode = max(after_sampled, key=after_sampled.get)
    summary = {
        "variant": str(spec["student_label"]),
        "target": str(spec["target"]),
        "weighting": str(spec["weighting"]),
        "teacher_temperature": float(spec["temperature"]),
        "harvest_successes": int(harvest.successes),
        "before_greedy_success": float(before_modes["greedy"]["overall"]["success_rate"]),
        "before_greedy_return": float(before_modes["greedy"]["overall"]["return_mean"]),
        "before_lower_success": float(before_modes["lower_t05"]["overall"]["success_rate"]),
        "before_gap_success": float(before_modes["gap_t055"]["overall"]["success_rate"]),
        "before_shoulder_success": float(before_modes["shoulder_t08"]["overall"]["success_rate"]),
        "before_best_sampled_mode": before_best_mode,
        "before_best_sampled_success": float(before_sampled[before_best_mode]),
        "after_greedy_success": float(after_modes["greedy"]["overall"]["success_rate"]),
        "after_greedy_return": float(after_modes["greedy"]["overall"]["return_mean"]),
        "after_lower_success": float(after_modes["lower_t05"]["overall"]["success_rate"]),
        "after_gap_success": float(after_modes["gap_t055"]["overall"]["success_rate"]),
        "after_shoulder_success": float(after_modes["shoulder_t08"]["overall"]["success_rate"]),
        "after_best_sampled_mode": after_best_mode,
        "after_best_sampled_success": float(after_sampled[after_best_mode]),
        **fine_tune_metrics,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Memory Next Practicalization Summary",
                "",
                f"- variant: `{summary['variant']}`",
                f"- family_key: `{spec['family_key']}`",
                f"- target: `{summary['target']}`",
                f"- weighting: `{summary['weighting']}`",
                f"- teacher_temperature: `{summary['teacher_temperature']}`",
                f"- before_greedy_success: `{summary['before_greedy_success']:.4f}`",
                f"- after_greedy_success: `{summary['after_greedy_success']:.4f}`",
                f"- before_best_sampled_mode: `{summary['before_best_sampled_mode']}`",
                f"- after_best_sampled_mode: `{summary['after_best_sampled_mode']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_stage0(campaign: dict[str, Any]) -> None:
    _write_stage0_reports(campaign)


def run_stage_a1(campaign: dict[str, Any], *, device: str) -> None:
    _run_task_queue(campaign, "source_manifest", _prepare_source_tasks(campaign), device=_resolved_device(device))
    _write_source_report(campaign)


def run_stage_a2(campaign: dict[str, Any]) -> None:
    _write_band_definition_report(campaign)


def run_stage_b1(campaign: dict[str, Any], *, device: str) -> None:
    _run_task_queue(campaign, "exploit_manifest", _prepare_exploit_tasks(campaign, phase="screening"), device=_resolved_device(device))
    _write_stage_b1_report(campaign)
    _run_task_queue(campaign, "exploit_manifest", _prepare_exploit_tasks(campaign, phase="rerun"), device=_resolved_device(device))
    _write_stage_b1_report(campaign)


def run_stage_c1(campaign: dict[str, Any], *, device: str) -> None:
    _run_task_queue(campaign, "explore_manifest", _prepare_practicalization_tasks(campaign, rerun=False), device=_resolved_device(device))
    _write_stage_c1_report(campaign)
    _run_task_queue(campaign, "explore_manifest", _prepare_practicalization_tasks(campaign, rerun=True), device=_resolved_device(device))
    _write_stage_c1_report(campaign)


def run_stage_d1(campaign: dict[str, Any]) -> None:
    _write_stage_d1_report(campaign)


def run_stage_d2(campaign: dict[str, Any], *, device: str) -> None:
    holdout_tasks = _prepare_holdout_control_tasks(campaign) + _prepare_explore_eval_tasks(campaign, phase="holdout")
    _run_task_queue(campaign, "holdout_manifest", holdout_tasks, device=_resolved_device(device))
    _write_stage_d2_report(campaign)


def run_stage_d3(campaign: dict[str, Any], *, device: str) -> None:
    healthy_tasks = _prepare_stage_d3_tasks(campaign, phase="healthy") + _prepare_explore_eval_tasks(campaign, phase="healthy")
    stability_tasks = _prepare_stage_d3_tasks(campaign, phase="stability") + _prepare_explore_eval_tasks(campaign, phase="stability")
    _run_task_queue(campaign, "healthy_manifest", healthy_tasks, device=_resolved_device(device))
    _run_task_queue(campaign, "verification_manifest", stability_tasks, device=_resolved_device(device))
    _write_stage_d3_report(campaign)


def run_stage_e1(campaign: dict[str, Any]) -> None:
    _payload, status = _write_candidate_pack_and_gate(campaign)
    _write_decision_memo(campaign, status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Large Memory conversion and practicalization campaign runner.")
    parser.add_argument(
        "--campaign-config",
        default="configs/experiments/lss_memory_next_program/campaign.yaml",
        help="Path to Memory-next campaign config",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("stage0")
    for name in ("stage-a1", "stage-b1", "stage-c1", "stage-d2", "stage-d3"):
        command = subparsers.add_parser(name)
        command.add_argument("--device", default="auto")
    subparsers.add_parser("stage-a2")
    subparsers.add_parser("stage-d1")
    subparsers.add_parser("stage-e1")

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
    if args.command == "stage0":
        run_stage0(campaign)
        return
    if args.command == "stage-a1":
        run_stage_a1(campaign, device=args.device)
        return
    if args.command == "stage-a2":
        run_stage_a2(campaign)
        return
    if args.command == "stage-b1":
        run_stage_b1(campaign, device=args.device)
        return
    if args.command == "stage-c1":
        run_stage_c1(campaign, device=args.device)
        return
    if args.command == "stage-d1":
        run_stage_d1(campaign)
        return
    if args.command == "stage-d2":
        run_stage_d2(campaign, device=args.device)
        return
    if args.command == "stage-d3":
        run_stage_d3(campaign, device=args.device)
        return
    if args.command == "stage-e1":
        run_stage_e1(campaign)
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
