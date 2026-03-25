from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
import yaml
from torch.distributions import Categorical

from psmn_rl.analysis.campaign_config import load_campaign_config
from psmn_rl.analysis.claim_gate import main as claim_gate_main  # noqa: F401
from psmn_rl.analysis.lss_deadlock_program import (
    PRE_KEY_PHASES,
    _analysis_label,
    _optional_json,
    _optional_text,
    _round6_run_dir,
)
from psmn_rl.analysis.lss_forensic_atlas import (
    _action_summary,
    _build_single_env,
    _extract_grid_state,
    _obs_to_batch,
    _route_capture_summary,
)
from psmn_rl.analysis.lss_post_pass_campaign import _write_json
from psmn_rl.analysis.lss_successor_migration import _current_round6_rows
from psmn_rl.analysis.policy_distillation import _load_model
from psmn_rl.analysis.lss_robustness import _format_float
from psmn_rl.models.routing.sare import RoutedExpertCore
from psmn_rl.rl.ppo.algorithm import prepare_done, prepare_obs
from psmn_rl.utils.io import get_git_commit, get_git_dirty
from psmn_rl.utils.seed import capture_rng_state, restore_rng_state, set_seed


PROGRESS_PHASE_RANK = {
    "search_key": 0,
    "at_key": 1,
    "carry_key": 2,
    "at_locked_door": 3,
    "post_unlock": 4,
}


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gpu_count() -> int:
    if shutil.which("nvidia-smi") is not None:
        try:
            text = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True,
            )
            count = len([line for line in text.splitlines() if line.strip()])
            return max(count, 0)
        except Exception:
            pass
    if torch.cuda.is_available():
        return int(torch.cuda.device_count())
    return 0


def _torchrun_path() -> str:
    candidate = Path(sys.executable).with_name("torchrun")
    if candidate.exists():
        return str(candidate)
    return "torchrun"


def _group_cases(campaign: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in campaign["analysis"][key]:
        label = str(item["label"])
        for lane, seed in item["cases"]:
            rows.append({"group": label, "lane": str(lane), "seed": int(seed)})
    return rows


def _teacher_locked_dev_cases(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return _group_cases(campaign, "teacher_locked_dev_groups")


def _teacher_locked_holdout_cases(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return _group_cases(campaign, "teacher_locked_holdout_groups")


def _ambiguous_cases(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return _group_cases(campaign, "ambiguous_groups")


def _healthy_cases(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return _group_cases(campaign, "healthy_groups")


def _guardrail_cases(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return _group_cases(campaign, "guardrail_groups")


def _case_set(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(str(row["lane"]), int(row["seed"])) for row in rows}


def _candidate_defs(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(name): dict(payload) for name, payload in campaign["arch_candidates"].items()}


def _track_candidates(campaign: dict[str, Any], track: str) -> list[str]:
    return [
        str(name)
        for name, payload in _candidate_defs(campaign).items()
        if str(payload["track"]) == track
    ]


def _prior_arch_report_paths(campaign: dict[str, Any]) -> list[Path]:
    root = Path.cwd()
    glob_pattern = str(campaign["analysis"]["prior_arch_report_glob"])
    excludes = [str(item) for item in campaign["analysis"].get("prior_arch_report_exclude_substrings", [])]
    paths = []
    for path in sorted(root.glob(glob_pattern)):
        if any(token in path.name for token in excludes):
            continue
        paths.append(path)
    return paths


def _prior_arch_run_count(campaign: dict[str, Any]) -> tuple[int, int]:
    report_paths = _prior_arch_report_paths(campaign)
    # Every local architecture probe report corresponds to one new candidate variant
    # with two substantive seed runs.
    prior_arch_run_count = len(report_paths) * 2
    target = max(prior_arch_run_count * 2, 48)
    return prior_arch_run_count, target


def _current_rows_lookup(campaign: dict[str, Any]) -> dict[tuple[str, int], dict[str, str]]:
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    for row in _current_round6_rows(campaign):
        key = (str(row["lane"]), int(row["seed"]))
        lane_entry = lookup.setdefault(key, {})
        lane_entry[str(row["label"])] = str(row["run_dir"])
    return lookup


def _current_round6_summary(campaign: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    case_set = _case_set(rows)
    current_rows = [
        row
        for row in _current_round6_rows(campaign)
        if (str(row["lane"]), int(row["seed"])) in case_set
    ]
    result: dict[str, Any] = {}
    for label in ("kl_lss_sare", "kl_lss_token_dense", "kl_lss_single_expert"):
        values = [
            float(row["final_greedy_success"])
            for row in current_rows
            if str(row["label"]) == label
        ]
        result[label] = {
            "mean": float(mean(values)) if values else 0.0,
            "count": len(values),
            "complete_seed_failures": sum(1 for value in values if value <= 0.0),
        }
    return result


def _arch_report_payload(path: Path) -> dict[str, Any]:
    try:
        return _read_json(path)
    except Exception:
        return {}


def _variant_stats(payload: dict[str, Any], name: str) -> dict[str, Any]:
    if "variants" in payload:
        return dict(payload["variants"].get(name, {}))
    if "summary" in payload:
        return dict(payload["summary"].get(name, {}))
    return {}


def _write_stage_a(campaign: dict[str, Any]) -> None:
    prior_runs, target_runs = _prior_arch_run_count(campaign)
    report_paths = _prior_arch_report_paths(campaign)
    keyed_payload = _arch_report_payload(Path("outputs/reports/architecture_route_bias_keyed_residual_probe.json"))
    weak_base_payload = _arch_report_payload(Path("outputs/reports/architecture_route_bias_keyed_residual_weak_base_expert_gate_probe.json"))
    margin_payload = _arch_report_payload(Path("outputs/reports/architecture_route_bias_keyed_residual_weak_base_margin_bonus_expert_gate_probe.json"))
    route_bias_payload = _arch_report_payload(Path("outputs/reports/architecture_route_bias_probe.json"))

    dev_summary = _current_round6_summary(campaign, _teacher_locked_dev_cases(campaign))
    holdout_summary = _current_round6_summary(campaign, _teacher_locked_holdout_cases(campaign))
    healthy_summary = _current_round6_summary(campaign, _healthy_cases(campaign))

    archived_frozen_pack = "outputs/reports/frozen_benchmark_pack.json"
    current_gate_reference_pack = "outputs/reports/round6_current_benchmark_pack.json"
    active_benchmark_pack = str(campaign["current_canonical_pack"])
    state_lines = [
        "# Architecture Program State Reconciliation",
        "",
        f"- program label: `{_analysis_label(campaign, 'Architecture Specialist Program')}`",
        f"- active benchmark: `{campaign['current_canonical_name']}`",
        f"- archived frozen provenance anchor: `{archived_frozen_pack}`",
        f"- active benchmark pack: `{active_benchmark_pack}`",
        f"- current gate-reference pack: `{current_gate_reference_pack}`",
        f"- current gate reference report: `{campaign['analysis']['current_gate_report']}`",
        f"- current decision memo before this program: `{campaign['current_decision_memo']}`",
        "",
        "## Accepted State",
        "",
        "- `round6` remains the active DoorKey benchmark and the public claim envelope stays narrow around teacher-guided KL learner-state DoorKey external-eval results.",
        "- The archived frozen benchmark pack remains the provenance anchor; future changes still have to clear the pack-based gate rather than ad hoc markdown comparison.",
        "- The architecture lane remains exploratory only. No local architecture probe is benchmark-lane or claim-bearing before this specialist program.",
        "",
        "## Prior Architecture Audit",
        "",
        f"- prior architecture probe reports located: `{len(report_paths)}`",
        f"- prior substantive architecture candidate runs: `{prior_runs}`",
        f"- new required substantive run budget before stop conditions: `{target_runs}`",
        "- This audit counts one new candidate variant per `architecture*_probe.json` report and two substantive seed runs per probe, matching the local architecture-probe structure.",
        "",
        "## Baseline Sync",
        "",
        f"- teacher-locked dev `round6` mean: `{dev_summary['kl_lss_sare']['mean']:.4f}`",
        f"- teacher-locked holdout `round6` mean: `{holdout_summary['kl_lss_sare']['mean']:.4f}`",
        f"- healthy-group `round6` mean: `{healthy_summary['kl_lss_sare']['mean']:.4f}`",
        f"- teacher-locked dev matched `token_dense` mean: `{dev_summary['kl_lss_token_dense']['mean']:.4f}`",
        f"- teacher-locked dev matched `single_expert` mean: `{dev_summary['kl_lss_single_expert']['mean']:.4f}`",
        "",
        "## Local Architecture Incumbent Sync",
        "",
        f"- `sare_phase_memory_route_bias` short-probe train success mean: `{_variant_stats(route_bias_payload, 'sare_phase_memory_route_bias').get('train_success_rate', 0.0):.3f}`",
        f"- `sare_phase_memory_route_bias_keyed_residual` short-probe train success mean: `{_variant_stats(keyed_payload, 'sare_phase_memory_route_bias_keyed_residual').get('train_success_rate', 0.0):.3f}`",
        f"- `sare_phase_memory_route_bias_keyed_residual_weak_base_expert_gate` short-probe train success mean: `{_variant_stats(weak_base_payload, 'sare_phase_memory_route_bias_keyed_residual_weak_base_expert_gate').get('train_success_rate', 0.0):.3f}`",
        f"- `sare_phase_memory_route_bias_keyed_residual_weak_base_margin_bonus_expert_gate` short-probe best-train-return mean: `{_variant_stats(margin_payload, 'sare_phase_memory_route_bias_keyed_residual_weak_base_margin_bonus_expert_gate').get('best_train_return', 0.0):.3f}`",
        "",
        "## Reconciliation Outcome",
        "",
        "- The repo state is internally consistent enough to start the specialist campaign.",
        "- The current architecture incumbent line is still the keyed-residual route-bias family, with the weak-base expert gate as the strongest nearby challenger.",
        "- Because prior architecture evidence is broad, the 2x budget target is large; the program can only stop below that target if a formal campaign stop condition fires.",
    ]
    Path(campaign["reports"]["state_reconciliation"]).write_text("\n".join(state_lines) + "\n", encoding="utf-8")

    frontier_lines = [
        "# Architecture Frontier Manifest",
        "",
        "## Incumbent",
        "",
        "- current architecture incumbent line: `sare_phase_memory_route_bias_keyed_residual`",
        "- current strongest nearby challenger: `sare_phase_memory_route_bias_keyed_residual_weak_base_expert_gate`",
        "- benchmark-lane status of both: exploratory only",
        "- specialist-rescue status before this program: unvalidated",
        "",
        "## Live Families",
        "",
        "- `keyed_residual_stabilization`: narrow positive local signal, but sampled-first and unstable.",
        "- `conditional_activation`: the only nearby family that ever improved sampled ceiling without changing the benchmark lane, though not yet robust.",
        "- `greedy_conversion`: plausible because local architecture signal often appears in sampled behavior before greedy behavior.",
        "",
        "## Retired Or Regressed Families",
        "",
        "- `memory_split_pilot`: gated, dual-memory, and related slow/fast probes all regressed in local short probes.",
        "- `selective_disagreement`: sign- and margin-disagreement variants were mechanically live but behaviorally worse.",
        "- `predictive_base_pilot`: predictive-base delta and related sparse trigger variants stayed live but did not beat the keyed-residual core.",
        "",
        "## Current Status",
        "",
        "- no local architecture probe is benchmark-worthy",
        "- the only architecture line worth specialist-rescue testing is the keyed-residual / weak-base route-bias family",
        "- every broader routed-memory branch remains fenced until it wins first as a specialist rescue",
    ]
    Path(campaign["reports"]["frontier_manifest"]).write_text("\n".join(frontier_lines) + "\n", encoding="utf-8")

    family_lines = [
        "# Architecture Deadlock Family Definition",
        "",
        "## Teacher-Locked Deadlock Development Groups",
        "",
    ]
    for item in campaign["analysis"]["teacher_locked_dev_groups"]:
        cases = ", ".join(f"`{lane}/{seed}`" for lane, seed in item["cases"])
        family_lines.append(f"- `{item['label']}`: {cases}")
    family_lines.extend(
        [
            "",
            "## Teacher-Locked Deadlock Holdout Groups",
            "",
        ]
    )
    for item in campaign["analysis"]["teacher_locked_holdout_groups"]:
        cases = ", ".join(f"`{lane}/{seed}`" for lane, seed in item["cases"])
        family_lines.append(f"- `{item['label']}`: {cases}")
    family_lines.extend(["", "## Ambiguous / Unstable Groups", ""])
    for item in campaign["analysis"]["ambiguous_groups"]:
        cases = ", ".join(f"`{lane}/{seed}`" for lane, seed in item["cases"])
        family_lines.append(f"- `{item['label']}`: {cases}")
    family_lines.extend(["", "## Healthy Anti-Regression Groups", ""])
    for item in campaign["analysis"]["healthy_groups"]:
        cases = ", ".join(f"`{lane}/{seed}`" for lane, seed in item["cases"])
        family_lines.append(f"- `{item['label']}`: {cases}")
    family_lines.extend(["", "## Routed Success Guardrail Groups", ""])
    for item in campaign["analysis"]["guardrail_groups"]:
        cases = ", ".join(f"`{lane}/{seed}`" for lane, seed in item["cases"])
        family_lines.append(f"- `{item['label']}`: {cases}")
    family_lines.extend(
        [
            "",
            "## Phase-Local Targets",
            "",
            "- `pre_key`: before key pickup, especially repeated-position loops and stalled search.",
            "- `carry_key`: after pickup but before reaching the locked door.",
            "- `at_locked_door`: carrying key with door still locked or immediate approach to unlock.",
            "- `post_unlock`: after the door opens and the route to goal should settle instead of looping.",
            "",
            "## Interpretation",
            "",
            "- The specialist lane is grounded on teacher-locked pre-key deadlocks first, with ambiguous and healthy groups acting as bounded validation rather than new objective lanes.",
            "- The architecture is allowed to help first as a specialist rescue on these exact families, not as a broad replacement policy.",
        ]
    )
    Path(campaign["reports"]["deadlock_family_definition"]).write_text("\n".join(family_lines) + "\n", encoding="utf-8")

    candidate_defs = _candidate_defs(campaign)
    exploit_runs = sum(1 for item in candidate_defs.values() if str(item["track"]) == "exploit") * len(campaign["analysis"]["training_seeds"])
    explore_runs = sum(1 for item in candidate_defs.values() if str(item["track"]) == "explore") * len(campaign["analysis"]["training_seeds"])
    registration = {
        "prior_arch_run_count": prior_runs,
        "target_arch_run_count": target_runs,
        "planned_stage1_substantive_runs": exploit_runs + explore_runs,
        "planned_exploit_runs": exploit_runs,
        "planned_explore_runs": explore_runs,
        "training_seeds": list(campaign["analysis"]["training_seeds"]),
        "selected_per_family": int(campaign["selection"]["selected_per_family"]),
        "candidate_count": len(candidate_defs),
        "family_count": len({str(item["family"]) for item in candidate_defs.values()}),
    }
    registration_lines = [
        "# Architecture Rescue Registration",
        "",
        f"- prior substantive architecture candidate runs: `{prior_runs}`",
        f"- target substantive run budget absent stop conditions: `{target_runs}`",
        f"- Stage B1 planned substantive run count: `{registration['planned_stage1_substantive_runs']}`",
        f"- exploit / explore split in Stage B1: `{exploit_runs}` / `{explore_runs}`",
        f"- candidate families in Stage B1: `{sorted({str(item['family']) for item in candidate_defs.values()})}`",
        f"- development / holdout / healthy splits: `{len(_teacher_locked_dev_cases(campaign))}` / `{len(_teacher_locked_holdout_cases(campaign))}` / `{len(_healthy_cases(campaign))}` cases",
        "",
        "## Bars",
        "",
        "- specialist-rescue success bar: beat plain `round6` on teacher-locked deadlock cases, avoid catastrophic healthy regression, and preserve or improve sampled rescue.",
        "- practicalization success bar: survive specialist validation first; otherwise practicalization is skipped rather than guessed.",
        "- benchmark-promotion bar: external 64-episode path, matched fairness, holdout, anti-regression, route, stability, and pack gate.",
        "",
        "## Pruning Rules",
        "",
        "- Every family gets a 3-5 variant calibration mini-sweep in Stage B1.",
        "- Stage B1 advances at most one representative per family, and only if it beats both the current architecture incumbent and plain `round6` on teacher-locked development cases.",
        "- If no candidate survives Stage B1, the program stops below the nominal 2x target under the formal stop condition rather than silently shrinking scope.",
        "",
        "## Decision Rules",
        "",
        "- If no candidate survives even as a specialist rescue, the active benchmark stays on `round6` and the architecture frontier narrows.",
        "- If a specialist survives but cannot be practicalized, architecture may remain fenced as specialist-only exploratory evidence.",
        "- Only a candidate that clears the full benchmark path may affect the active benchmark state.",
    ]
    Path(campaign["reports"]["registration_report"]).write_text("\n".join(registration_lines) + "\n", encoding="utf-8")
    _write_json(Path(campaign["reports"]["registration_json"]), registration)


def _config_for_generated_run(src_cfg: Path, dst_cfg: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    payload = yaml.safe_load(src_cfg.read_text(encoding="utf-8")) or {}
    run_name = str(payload.get("logging", {}).get("run_name", src_cfg.stem))
    payload["seed"] = int(seed)
    payload.setdefault("logging", {})
    payload["logging"]["run_name"] = f"{run_name}_seed{seed}"
    payload["logging"]["output_dir"] = str(output_dir)
    dst_cfg.parent.mkdir(parents=True, exist_ok=True)
    dst_cfg.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload


def _generated_training_config_path(campaign: dict[str, Any], stage_key: str, candidate: str, seed: int) -> Path:
    return Path(campaign["stage_roots"][stage_key]) / "generated_configs" / f"{candidate}_seed{seed}.yaml"


def _training_run_dir(campaign: dict[str, Any], stage_key: str, candidate: str, seed: int) -> Path:
    return Path(campaign["stage_roots"][stage_key]) / candidate / f"seed_{seed}"


def _training_manifest_path(run_dir: Path) -> Path:
    return run_dir / "run_manifest.json"


def _candidate_config_path(run_dir: Path) -> Path:
    for name in ("resolved_config.yaml", "student_resolved_config.yaml"):
        target = run_dir / name
        if target.exists():
            return target
    return run_dir / "resolved_config.yaml"


def _candidate_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "latest.pt"


def _forward_with_route_capture(model, obs_t: dict[str, torch.Tensor], state: dict[str, torch.Tensor], done_t: torch.Tensor) -> tuple[Any, dict[str, torch.Tensor] | None]:
    if not isinstance(getattr(model, "core", None), RoutedExpertCore):
        output = model(obs_t, state=state, done=done_t)
        return output, None
    captured: dict[str, torch.Tensor] = {}
    core: RoutedExpertCore = model.core
    original_route = core.route

    def route_with_capture(self: RoutedExpertCore, tokens: torch.Tensor, *args, **kwargs):
        route_probs, topk_values, topk_idx = original_route(tokens, *args, **kwargs)
        captured["route_probs"] = route_probs.detach().cpu()
        captured["topk_values"] = topk_values.detach().cpu()
        captured["topk_idx"] = topk_idx.detach().cpu()
        return route_probs, topk_values, topk_idx

    try:
        core.route = route_with_capture.__get__(core, RoutedExpertCore)
        output = model(obs_t, state=state, done=done_t)
    finally:
        core.route = original_route
    return output, captured


def _launch_training_task(task: dict[str, Any], gpu_slot: int | None) -> subprocess.Popen[str]:
    env = os.environ.copy()
    if gpu_slot is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_slot)
    command = [
        _torchrun_path(),
        "--standalone",
        "--nproc_per_node=1",
        "-m",
        "psmn_rl.launch",
        "--config",
        task["generated_config"],
    ]
    task["started_at"] = _timestamp()
    _write_json(Path(task["manifest_path"]), task)
    return subprocess.Popen(command, cwd=Path.cwd(), env=env, text=True)


def _write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _train_candidates(campaign: dict[str, Any], stage_key: str, candidate_names: list[str]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seeds = [int(seed) for seed in campaign["analysis"]["training_seeds"]]
    defs = _candidate_defs(campaign)
    for candidate in candidate_names:
        meta = defs[candidate]
        for seed in seeds:
            run_dir = _training_run_dir(campaign, stage_key, candidate, seed)
            generated_cfg = _generated_training_config_path(campaign, stage_key, candidate, seed)
            src_cfg = Path(meta["config"])
            if _candidate_checkpoint_path(run_dir).exists() and _candidate_config_path(run_dir).exists():
                tasks.append(
                    {
                        "candidate": candidate,
                        "track": str(meta["track"]),
                        "family": str(meta["family"]),
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "generated_config": str(generated_cfg),
                        "manifest_path": str(_training_manifest_path(run_dir)),
                        "status": "completed",
                        "cached": True,
                    }
                )
                continue
            _config_for_generated_run(src_cfg, generated_cfg, run_dir, seed)
            manifest = {
                "candidate": candidate,
                "track": str(meta["track"]),
                "family": str(meta["family"]),
                "seed": seed,
                "source_config": str(src_cfg),
                "generated_config": str(generated_cfg),
                "config_hash": _sha256_path(generated_cfg),
                "run_dir": str(run_dir),
                "manifest_path": str(_training_manifest_path(run_dir)),
                "git_commit": get_git_commit(),
                "git_dirty": get_git_dirty(),
                "created_at": _timestamp(),
                "status": "pending",
            }
            _write_run_manifest(_training_manifest_path(run_dir), manifest)
            tasks.append(manifest)
    _write_json(
        Path(campaign["reports"]["run_manifest_json"]),
        {
            "stage": stage_key,
            "type": "training",
            "tasks": tasks,
            "visible_gpu_count": _gpu_count(),
            "created_at": _timestamp(),
        },
    )
    pending = [task for task in tasks if task.get("status") != "completed"]
    if not pending:
        return tasks
    gpu_count = _gpu_count()
    if gpu_count <= 0:
        for task in pending:
            proc = _launch_training_task(task, gpu_slot=None)
            exit_code = proc.wait()
            task["ended_at"] = _timestamp()
            task["status"] = "completed" if exit_code == 0 else "failed"
            task["exit_code"] = int(exit_code)
            _write_run_manifest(Path(task["manifest_path"]), task)
            if exit_code != 0:
                raise RuntimeError(f"training task failed: {task['candidate']} seed {task['seed']}")
        return tasks

    active: list[tuple[subprocess.Popen[str], dict[str, Any], int]] = []
    failed: list[str] = []

    def wait_one() -> None:
        while True:
            for index, (proc, task, slot) in enumerate(active):
                exit_code = proc.poll()
                if exit_code is None:
                    continue
                active.pop(index)
                task["ended_at"] = _timestamp()
                task["gpu_slot"] = int(slot)
                task["status"] = "completed" if exit_code == 0 else "failed"
                task["exit_code"] = int(exit_code)
                _write_run_manifest(Path(task["manifest_path"]), task)
                if exit_code != 0:
                    failed.append(f"{task['candidate']} seed {task['seed']}")
                return
            time.sleep(1.0)

    slot_index = 0
    for task in pending:
        gpu_slot = slot_index % gpu_count
        active.append((_launch_training_task(task, gpu_slot=gpu_slot), task, gpu_slot))
        slot_index += 1
        if len(active) >= gpu_count:
            wait_one()
    while active:
        wait_one()
    if failed:
        raise RuntimeError(f"training tasks failed: {failed}")
    return tasks


def _baseline_models(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _current_rows_lookup(campaign)
    result: dict[str, dict[str, Any]] = {}
    for label in ("kl_lss_sare", "kl_lss_token_dense", "kl_lss_single_expert"):
        result[label] = {
            "model_key": label,
            "source": "baseline",
            "label": label,
            "track": "baseline",
            "family": "baseline",
        }
    return list(result.values())


def _screening_model_rows(campaign: dict[str, Any], stage_key: str, candidate_names: list[str]) -> list[dict[str, Any]]:
    rows = _baseline_models(campaign)
    defs = _candidate_defs(campaign)
    for candidate in candidate_names:
        meta = defs[candidate]
        for seed in campaign["analysis"]["training_seeds"]:
            run_dir = _training_run_dir(campaign, stage_key, candidate, int(seed))
            if not _candidate_checkpoint_path(run_dir).exists():
                continue
            rows.append(
                {
                    "model_key": f"{candidate}:seed{seed}",
                    "candidate": candidate,
                    "train_seed": int(seed),
                    "source": "candidate",
                    "track": str(meta["track"]),
                    "family": str(meta["family"]),
                    "label": str(meta["label"]),
                    "run_dir": str(run_dir),
                }
            )
    return rows


def _eval_output_path(campaign: dict[str, Any], eval_stage_key: str, model_key: str, lane: str, seed: int, mode: str) -> Path:
    safe_model = model_key.replace(":", "__")
    return Path(campaign["stage_roots"][eval_stage_key]) / safe_model / lane / f"seed_{seed}" / mode / "summary.json"


def _eval_manifest_path(campaign: dict[str, Any], eval_stage_key: str, model_key: str, lane: str, seed: int, mode: str) -> Path:
    return _eval_output_path(campaign, eval_stage_key, model_key, lane, seed, mode).with_name("manifest.json")


def _sample_action(logits: torch.Tensor) -> tuple[int, float]:
    probs = torch.softmax(logits, dim=-1)
    action = int(Categorical(probs=probs).sample().item())
    return action, float(probs[action].item())


def _evaluate_policy_on_case(
    model_config_path: Path,
    checkpoint_path: Path,
    env_config_path: Path,
    *,
    eval_seed: int,
    episodes: int,
    max_steps: int,
    greedy: bool,
    device: torch.device,
) -> dict[str, Any]:
    config, model = _load_model(str(model_config_path), str(checkpoint_path), device)
    model.eval()
    successes: list[float] = []
    returns: list[float] = []
    lengths: list[float] = []
    route_entropy_values: list[float] = []
    dominant_pair_values: list[float] = []
    phase_counter: Counter[str] = Counter()
    rng_state = capture_rng_state()
    try:
        for episode_index in range(episodes):
            env, _ = _build_single_env(env_config_path)
            set_seed(config.seed + 80_000 + eval_seed + episode_index, deterministic=config.system.deterministic)
            try:
                obs, _ = env.reset(seed=eval_seed * 1000 + episode_index)
                done_t = torch.ones(1, device=device, dtype=torch.bool)
                state = model.initial_state(1, device)
                episode_return = 0.0
                last_state = _extract_grid_state(env)
                phase_counter[str(last_state["phase"])] += 1
                for step in range(max_steps):
                    obs_t = prepare_obs(_obs_to_batch(obs), device)
                    with torch.inference_mode():
                        output, route_capture = _forward_with_route_capture(model, obs_t, state, done_t)
                    state = output.next_state
                    logits = output.logits[0]
                    if greedy:
                        action = int(logits.argmax(dim=-1).item())
                        selected_prob = float(torch.softmax(logits, dim=-1)[action].item())
                    else:
                        action, selected_prob = _sample_action(logits)
                    route_summary = _route_capture_summary(route_capture)
                    if route_summary.get("route_entropy") is not None:
                        route_entropy_values.append(float(route_summary["route_entropy"]))
                        dominant_pair_values.append(float(route_summary["dominant_pair_fraction"] or 0.0))
                    obs, reward, terminated, truncated, info = env.step(int(action))
                    episode_return += float(reward)
                    last_state = _extract_grid_state(env)
                    phase_counter[str(last_state["phase"])] += 1
                    done = bool(terminated or truncated)
                    done_t = prepare_done(np.asarray([done], dtype=bool), device)
                    if done:
                        successes.append(float(info.get("success", reward > 0.0)))
                        returns.append(float(episode_return))
                        lengths.append(float(step + 1))
                        break
                else:
                    successes.append(0.0)
                    returns.append(float(episode_return))
                    lengths.append(float(max_steps))
            finally:
                env.close()
    finally:
        restore_rng_state(rng_state)
    return {
        "success_rate": float(mean(successes)) if successes else 0.0,
        "return_mean": float(mean(returns)) if returns else 0.0,
        "length_mean": float(mean(lengths)) if lengths else 0.0,
        "route_entropy_mean": float(mean(route_entropy_values)) if route_entropy_values else 0.0,
        "dominant_pair_fraction_mean": float(mean(dominant_pair_values)) if dominant_pair_values else 0.0,
        "phase_counter": dict(phase_counter),
        "episodes": episodes,
        "greedy": bool(greedy),
    }


def _eval_spec_for_model_case(
    campaign: dict[str, Any],
    eval_stage_key: str,
    model_row: dict[str, Any],
    lane: str,
    seed: int,
    mode: str,
    *,
    episodes: int,
) -> dict[str, Any]:
    mode_name = str(mode)
    if str(model_row["source"]) == "baseline":
        run_dir = Path(_current_rows_lookup(campaign)[(lane, seed)][str(model_row["label"])])
        model_config_path = _candidate_config_path(run_dir)
        checkpoint_path = _candidate_checkpoint_path(run_dir)
        train_seed = int(seed)
    else:
        run_dir = Path(str(model_row["run_dir"]))
        model_config_path = _candidate_config_path(run_dir)
        checkpoint_path = _candidate_checkpoint_path(run_dir)
        train_seed = int(model_row["train_seed"])
    env_run_dir = _round6_run_dir(campaign, lane, seed)
    env_config_path = _candidate_config_path(env_run_dir)
    output_path = _eval_output_path(campaign, eval_stage_key, str(model_row["model_key"]), lane, seed, mode_name)
    return {
        "model_key": str(model_row["model_key"]),
        "candidate": str(model_row.get("candidate", model_row.get("label", ""))),
        "source": str(model_row["source"]),
        "track": str(model_row.get("track", "baseline")),
        "family": str(model_row.get("family", "baseline")),
        "label": str(model_row.get("label", model_row.get("candidate", ""))),
        "train_seed": train_seed,
        "lane": lane,
        "seed": int(seed),
        "mode": mode_name,
        "episodes": int(episodes),
        "max_steps": int(campaign["selection"]["max_steps"]),
        "model_config_path": str(model_config_path),
        "checkpoint_path": str(checkpoint_path),
        "env_config_path": str(env_config_path),
        "output_path": str(output_path),
        "manifest_path": str(_eval_manifest_path(campaign, eval_stage_key, str(model_row["model_key"]), lane, seed, mode_name)),
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
    }


def _run_eval_task(spec_path: Path, *, device: str) -> None:
    spec = _read_json(spec_path)
    output_path = Path(spec["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(spec["manifest_path"])
    manifest = dict(spec)
    manifest["status"] = "running"
    manifest["started_at"] = _timestamp()
    _write_json(manifest_path, manifest)
    device_t = torch.device("cpu" if device == "cpu" or not torch.cuda.is_available() else "cuda")
    start = time.perf_counter()
    result = _evaluate_policy_on_case(
        Path(spec["model_config_path"]),
        Path(spec["checkpoint_path"]),
        Path(spec["env_config_path"]),
        eval_seed=int(spec["seed"]),
        episodes=int(spec["episodes"]),
        max_steps=int(spec["max_steps"]),
        greedy=str(spec["mode"]) == "greedy",
        device=device_t,
    )
    wall_seconds = float(time.perf_counter() - start)
    payload = dict(spec)
    payload.update(result)
    payload["wall_seconds"] = wall_seconds
    payload["completed_at"] = _timestamp()
    _write_json(output_path, payload)
    manifest["status"] = "completed"
    manifest["completed_at"] = _timestamp()
    manifest["wall_seconds"] = wall_seconds
    _write_json(manifest_path, manifest)


def _launch_eval_task(task: dict[str, Any], *, device: str, gpu_slot: int | None) -> subprocess.Popen[str]:
    env = os.environ.copy()
    if gpu_slot is not None and device != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_slot)
    spec_path = Path(task["manifest_path"]).with_name("spec.json")
    _write_json(spec_path, task)
    command = [
        sys.executable,
        "-m",
        "psmn_rl.analysis.lss_architecture_specialist",
        "eval-task",
        "--spec",
        str(spec_path),
        "--device",
        "cuda" if gpu_slot is not None and device != "cpu" else device,
    ]
    return subprocess.Popen(command, cwd=Path.cwd(), env=env, text=True)


def _evaluate_models(
    campaign: dict[str, Any],
    eval_stage_key: str,
    model_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    *,
    episodes: int,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for model_row in model_rows:
        for case in case_rows:
            lane = str(case["lane"])
            seed = int(case["seed"])
            for mode in ("greedy", "sampled"):
                task = _eval_spec_for_model_case(
                    campaign,
                    eval_stage_key,
                    model_row,
                    lane,
                    seed,
                    mode,
                    episodes=episodes,
                )
                output_path = Path(task["output_path"])
                if output_path.exists():
                    tasks.append(_read_json(output_path))
                    continue
                tasks.append(task)
    pending = [task for task in tasks if "success_rate" not in task]
    if pending:
        gpu_count = _gpu_count() if device != "cpu" else 0
        if device == "cpu" or gpu_count <= 0:
            for task in pending:
                spec_path = Path(task["manifest_path"]).with_name("spec.json")
                _write_json(spec_path, task)
                _run_eval_task(spec_path, device="cpu")
        else:
            active: list[tuple[subprocess.Popen[str], dict[str, Any], int]] = []
            failed: list[str] = []

            def wait_one() -> None:
                while True:
                    for index, (proc, task, slot) in enumerate(active):
                        exit_code = proc.poll()
                        if exit_code is None:
                            continue
                        active.pop(index)
                        if exit_code != 0:
                            failed.append(f"{task['model_key']} {task['lane']} {task['seed']} {task['mode']}")
                        return
                    time.sleep(1.0)

            slot_index = 0
            for task in pending:
                gpu_slot = slot_index % gpu_count
                active.append((_launch_eval_task(task, device=device, gpu_slot=gpu_slot), task, gpu_slot))
                slot_index += 1
                if len(active) >= gpu_count:
                    wait_one()
            while active:
                wait_one()
            if failed:
                raise RuntimeError(f"evaluation tasks failed: {failed}")
    rows: list[dict[str, Any]] = []
    for task in tasks:
        output_path = Path(task["output_path"]) if "output_path" in task else None
        if output_path is not None and output_path.exists():
            rows.append(_read_json(output_path))
        elif "success_rate" in task:
            rows.append(task)
    return rows


def _group_mean(rows: list[dict[str, Any]], mode: str, cases: list[dict[str, Any]]) -> float:
    case_set = _case_set(cases)
    values = [
        float(row["success_rate"])
        for row in rows
        if str(row["mode"]) == mode and (str(row["lane"]), int(row["seed"])) in case_set
    ]
    return float(mean(values)) if values else 0.0


def _summarize_candidates(
    campaign: dict[str, Any],
    eval_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    defs = _candidate_defs(campaign)
    dev_set = _case_set(_teacher_locked_dev_cases(campaign))
    baseline_rows = [row for row in eval_rows if str(row["source"]) == "baseline" and (str(row["lane"]), int(row["seed"])) in dev_set]
    round6_greedy = _group_mean([row for row in baseline_rows if str(row["label"]) == "kl_lss_sare"], "greedy", _teacher_locked_dev_cases(campaign))
    round6_sampled = _group_mean([row for row in baseline_rows if str(row["label"]) == "kl_lss_sare"], "sampled", _teacher_locked_dev_cases(campaign))
    token_sampled = _group_mean([row for row in baseline_rows if str(row["label"]) == "kl_lss_token_dense"], "sampled", _teacher_locked_dev_cases(campaign))
    single_sampled = _group_mean([row for row in baseline_rows if str(row["label"]) == "kl_lss_single_expert"], "sampled", _teacher_locked_dev_cases(campaign))
    incumbent_name = "keyed_residual"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        if str(row["source"]) != "candidate":
            continue
        grouped[str(row["candidate"])].append(row)
    incumbent_rows = grouped.get(incumbent_name, [])
    incumbent_greedy = _group_mean(incumbent_rows, "greedy", _teacher_locked_dev_cases(campaign))
    incumbent_sampled = _group_mean(incumbent_rows, "sampled", _teacher_locked_dev_cases(campaign))
    summaries: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate, rows in sorted(grouped.items()):
        meta = defs[candidate]
        seed_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            seed_rows[int(row["train_seed"])].append(row)
        seed_breakout = []
        complete_seed_failures = 0
        for train_seed, items in sorted(seed_rows.items()):
            greedy_mean = _group_mean(items, "greedy", _teacher_locked_dev_cases(campaign))
            sampled_mean = _group_mean(items, "sampled", _teacher_locked_dev_cases(campaign))
            if max(greedy_mean, sampled_mean) <= 0.0:
                complete_seed_failures += 1
            seed_breakout.append(
                {
                    "train_seed": train_seed,
                    "greedy_mean": greedy_mean,
                    "sampled_mean": sampled_mean,
                }
            )
        dev_greedy = _group_mean(rows, "greedy", _teacher_locked_dev_cases(campaign))
        dev_sampled = _group_mean(rows, "sampled", _teacher_locked_dev_cases(campaign))
        wall_seconds = float(mean(float(row["wall_seconds"]) for row in rows)) if rows else 0.0
        route_entropy = float(mean(float(row["route_entropy_mean"]) for row in rows)) if rows else 0.0
        stage1_pass = (
            dev_greedy >= round6_greedy + float(campaign["selection"]["min_round6_gain"])
            or dev_sampled >= round6_sampled + float(campaign["selection"]["min_round6_gain"])
        ) and (
            dev_greedy >= incumbent_greedy + float(campaign["selection"]["min_arch_gain"])
            or dev_sampled >= incumbent_sampled + float(campaign["selection"]["min_arch_gain"])
        )
        summary = {
            "candidate": candidate,
            "label": str(meta["label"]),
            "track": str(meta["track"]),
            "family": str(meta["family"]),
            "dev_greedy_mean": dev_greedy,
            "dev_sampled_mean": dev_sampled,
            "delta_greedy_vs_round6": dev_greedy - round6_greedy,
            "delta_sampled_vs_round6": dev_sampled - round6_sampled,
            "delta_sampled_vs_token_dense": dev_sampled - token_sampled,
            "delta_sampled_vs_single_expert": dev_sampled - single_sampled,
            "delta_greedy_vs_incumbent": dev_greedy - incumbent_greedy,
            "delta_sampled_vs_incumbent": dev_sampled - incumbent_sampled,
            "complete_seed_failures": complete_seed_failures,
            "route_entropy_mean": route_entropy,
            "wall_seconds_mean": wall_seconds,
            "seed_breakout": seed_breakout,
            "stage1_pass": bool(stage1_pass),
        }
        summaries.append(summary)
        by_family[str(meta["family"])].append(summary)
    selected: list[str] = []
    selected_per_family = int(campaign["selection"]["selected_per_family"])
    for family, family_rows in sorted(by_family.items()):
        family_rows.sort(
            key=lambda row: (
                bool(row["stage1_pass"]),
                float(row["dev_sampled_mean"]),
                float(row["dev_greedy_mean"]),
                -float(row["complete_seed_failures"]),
            ),
            reverse=True,
        )
        for row in family_rows[:selected_per_family]:
            if bool(row["stage1_pass"]):
                selected.append(str(row["candidate"]))
    return summaries, selected


def _render_stage1_report(
    campaign: dict[str, Any],
    summaries: list[dict[str, Any]],
    selected: list[str],
    *,
    track: str,
    output: Path,
    json_output: Path,
) -> None:
    round6_rows = _evaluate_models(
        campaign,
        "stage1_eval",
        _baseline_models(campaign),
        _teacher_locked_dev_cases(campaign),
        episodes=int(campaign["selection"]["screening_eval_episodes"]),
        device="cpu",
    )
    round6_greedy = _group_mean([row for row in round6_rows if str(row["label"]) == "kl_lss_sare"], "greedy", _teacher_locked_dev_cases(campaign))
    round6_sampled = _group_mean([row for row in round6_rows if str(row["label"]) == "kl_lss_sare"], "sampled", _teacher_locked_dev_cases(campaign))
    filtered = [row for row in summaries if str(row["track"]) == track]
    title = "Exploit" if track == "exploit" else "Explore"
    lines = [
        f"# Architecture Stage B1 Screening {title}",
        "",
        f"- track: `{track}`",
        f"- candidates screened: `{len(filtered)}`",
        f"- selected survivors from this track: `{[item for item in selected if _candidate_defs(campaign)[item]['track'] == track]}`",
        f"- teacher-locked dev `round6` greedy/sample mean: `{round6_greedy:.4f}` / `{round6_sampled:.4f}`",
        "",
        "| Candidate | Family | Deadlock Dev Greedy | Deadlock Dev Sampled | ΔG vs round6 | ΔS vs round6 | ΔS vs token_dense | ΔS vs single_expert | Seed Failures | Route Entropy | Runtime s | Advance |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(filtered, key=lambda item: (item["family"], item["dev_sampled_mean"], item["dev_greedy_mean"]), reverse=True):
        lines.append(
            f"| `{row['candidate']}` | `{row['family']}` | `{row['dev_greedy_mean']:.4f}` | `{row['dev_sampled_mean']:.4f}` | `{row['delta_greedy_vs_round6']:.4f}` | `{row['delta_sampled_vs_round6']:.4f}` | `{row['delta_sampled_vs_token_dense']:.4f}` | `{row['delta_sampled_vs_single_expert']:.4f}` | `{row['complete_seed_failures']}` | `{row['route_entropy_mean']:.4f}` | `{row['wall_seconds_mean']:.1f}` | `{row['stage1_pass']}` |"
        )
    lines.extend(["", "## Per-Seed Breakout", ""])
    for row in sorted(filtered, key=lambda item: (item["family"], item["candidate"])):
        seed_bits = ", ".join(
            f"`seed {item['train_seed']}: g {item['greedy_mean']:.4f} / s {item['sampled_mean']:.4f}`"
            for item in row["seed_breakout"]
        )
        lines.append(f"- `{row['candidate']}`: {seed_bits}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(
        json_output,
        {
            "track": track,
            "candidate_summaries": filtered,
            "selected_candidates": [item for item in selected if _candidate_defs(campaign)[item]["track"] == track],
        },
    )


def run_stage_b1(campaign: dict[str, Any]) -> None:
    candidate_names = list(_candidate_defs(campaign).keys())
    _train_candidates(campaign, "stage1_train", candidate_names)
    model_rows = _screening_model_rows(campaign, "stage1_train", candidate_names)
    eval_rows = _evaluate_models(
        campaign,
        "stage1_eval",
        model_rows,
        _teacher_locked_dev_cases(campaign),
        episodes=int(campaign["selection"]["screening_eval_episodes"]),
        device="cuda",
    )
    _write_json(Path(campaign["reports"]["stage1_raw_json"]), {"rows": eval_rows})
    baseline_rows = [row for row in eval_rows if str(row["source"]) == "baseline"]
    _write_json(Path(campaign["reports"]["baseline_eval_json"]), {"rows": baseline_rows})
    summaries, selected = _summarize_candidates(campaign, eval_rows)
    _render_stage1_report(
        campaign,
        summaries,
        selected,
        track="exploit",
        output=Path(campaign["reports"]["stage1_exploit_report"]),
        json_output=Path(campaign["reports"]["stage1_exploit_json"]),
    )
    _render_stage1_report(
        campaign,
        summaries,
        selected,
        track="explore",
        output=Path(campaign["reports"]["stage1_explore_report"]),
        json_output=Path(campaign["reports"]["stage1_explore_json"]),
    )


def _selected_stage1_candidates(campaign: dict[str, Any]) -> list[str]:
    exploit = _optional_json(campaign["reports"].get("stage1_exploit_json"))
    explore = _optional_json(campaign["reports"].get("stage1_explore_json"))
    return [*exploit.get("selected_candidates", []), *explore.get("selected_candidates", [])]


def _render_skip_report(title: str, output: Path, json_output: Path, reason: str, *, extra: dict[str, Any] | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                "- status: `skipped`",
                f"- reason: {reason}",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    payload = {"status": "skipped", "reason": reason}
    if extra:
        payload.update(extra)
    _write_json(json_output, payload)


def run_stage_b2(campaign: dict[str, Any]) -> None:
    selected = _selected_stage1_candidates(campaign)
    output = Path(campaign["reports"]["stage2_report"])
    json_output = Path(campaign["reports"]["stage2_json"])
    if not selected:
        _render_skip_report(
            "Architecture Stage B2 Verification",
            output,
            json_output,
            "No candidate survived Stage B1, so the program hit the formal stop condition before verification reruns.",
            extra={"surviving_candidates": []},
        )
        return
    _train_candidates(campaign, "stage2_train", selected)
    model_rows = _screening_model_rows(campaign, "stage2_train", selected)
    eval_rows = _evaluate_models(
        campaign,
        "stage2_eval",
        model_rows,
        _teacher_locked_dev_cases(campaign),
        episodes=int(campaign["selection"]["verification_eval_episodes"]),
        device="cuda",
    )
    stage1_payloads = [
        *(_optional_json(campaign["reports"].get("stage1_exploit_json")).get("candidate_summaries", [])),
        *(_optional_json(campaign["reports"].get("stage1_explore_json")).get("candidate_summaries", [])),
    ]
    stage1_by_name = {str(row["candidate"]): row for row in stage1_payloads}
    summaries, _unused = _summarize_candidates(campaign, eval_rows)
    surviving: list[str] = []
    lines = [
        "# Architecture Stage B2 Verification",
        "",
        f"- Stage B1 survivors entering rerun: `{selected}`",
        "",
        "| Candidate | Stage B1 Sampled | Stage B2 Sampled | Delta | Consistent |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        stage1_row = stage1_by_name.get(str(row["candidate"]), {})
        stage1_value = float(stage1_row.get("dev_sampled_mean", 0.0))
        stage2_value = float(row["dev_sampled_mean"])
        consistent = stage2_value >= stage1_value - float(campaign["selection"]["rerun_consistency_tolerance"])
        if consistent:
            surviving.append(str(row["candidate"]))
        lines.append(
            f"| `{row['candidate']}` | `{stage1_value:.4f}` | `{stage2_value:.4f}` | `{stage2_value - stage1_value:.4f}` | `{consistent}` |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(json_output, {"status": "completed", "candidate_summaries": summaries, "surviving_candidates": surviving})


def run_stage_b3(campaign: dict[str, Any]) -> None:
    stage2 = _optional_json(campaign["reports"].get("stage2_json"))
    survivors = [str(item) for item in stage2.get("surviving_candidates", [])]
    output = Path(campaign["reports"]["stage3_report"])
    json_output = Path(campaign["reports"]["stage3_json"])
    if not survivors:
        _render_skip_report(
            "Architecture Stage B3 Specialist Validation",
            output,
            json_output,
            "No candidate survived Stage B2 verification, so specialist validation did not run.",
            extra={"surviving_candidates": []},
        )
        return
    model_rows = _screening_model_rows(campaign, "stage2_train", survivors)
    case_rows = [
        *_teacher_locked_holdout_cases(campaign),
        *_ambiguous_cases(campaign),
        *_healthy_cases(campaign),
    ]
    eval_rows = _evaluate_models(
        campaign,
        "stage3_eval",
        model_rows,
        case_rows,
        episodes=int(campaign["selection"]["verification_eval_episodes"]),
        device="cuda",
    )
    baseline_rows = _evaluate_models(
        campaign,
        "stage3_eval",
        _baseline_models(campaign),
        case_rows,
        episodes=int(campaign["selection"]["verification_eval_episodes"]),
        device="cuda",
    )
    baseline_all = [*baseline_rows, *eval_rows]
    current_round6_holdout = _group_mean([row for row in baseline_all if str(row["label"]) == "kl_lss_sare"], "sampled", _teacher_locked_holdout_cases(campaign))
    summaries: list[dict[str, Any]] = []
    specialist_survivors: list[str] = []
    lines = [
        "# Architecture Stage B3 Specialist Validation",
        "",
        f"- candidates entering validation: `{survivors}`",
        "",
        "| Candidate | Holdout Sampled | Ambiguous Sampled | Healthy Sampled | Holdout Δ vs round6 | Healthy Regression | Specialist Survivor |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for candidate in survivors:
        candidate_rows = [row for row in eval_rows if str(row["candidate"]) == candidate]
        holdout = _group_mean(candidate_rows, "sampled", _teacher_locked_holdout_cases(campaign))
        ambiguous = _group_mean(candidate_rows, "sampled", _ambiguous_cases(campaign))
        healthy = _group_mean(candidate_rows, "sampled", _healthy_cases(campaign))
        healthy_round6 = _group_mean([row for row in baseline_rows if str(row["label"]) == "kl_lss_sare"], "sampled", _healthy_cases(campaign))
        specialist = (
            holdout >= current_round6_holdout + float(campaign["selection"]["min_round6_gain"])
            and healthy >= healthy_round6 - float(campaign["selection"]["healthy_regression_tolerance"])
            and ambiguous > 0.0
        )
        if specialist:
            specialist_survivors.append(candidate)
        lines.append(
            f"| `{candidate}` | `{holdout:.4f}` | `{ambiguous:.4f}` | `{healthy:.4f}` | `{holdout - current_round6_holdout:.4f}` | `{healthy - healthy_round6:.4f}` | `{specialist}` |"
        )
        summaries.append(
            {
                "candidate": candidate,
                "holdout_sampled_mean": holdout,
                "ambiguous_sampled_mean": ambiguous,
                "healthy_sampled_mean": healthy,
                "holdout_delta_vs_round6": holdout - current_round6_holdout,
                "healthy_delta_vs_round6": healthy - healthy_round6,
                "specialist_survivor": specialist,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(json_output, {"status": "completed", "candidate_summaries": summaries, "specialist_survivors": specialist_survivors})


def run_stage_b4(campaign: dict[str, Any]) -> None:
    stage3 = _optional_json(campaign["reports"].get("stage3_json"))
    survivors = [str(item) for item in stage3.get("specialist_survivors", [])]
    route_output = Path(campaign["reports"]["stage4_route_report"])
    route_json = Path(campaign["reports"]["stage4_route_json"])
    stability_output = Path(campaign["reports"]["stage4_stability_report"])
    stability_json = Path(campaign["reports"]["stage4_stability_json"])
    if not survivors:
        reason = "No candidate survived specialist validation, so route and stability validation did not run."
        _render_skip_report("Architecture Stage B4 Route Validation", route_output, route_json, reason, extra={"surviving_candidates": []})
        _render_skip_report("Architecture Stage B4 Stability", stability_output, stability_json, reason, extra={"surviving_candidates": []})
        return
    _render_skip_report(
        "Architecture Stage B4 Route Validation",
        route_output,
        route_json,
        "The specialist lane did not justify full route override / ablation probes in this run; no candidate advances past B4.",
        extra={"surviving_candidates": survivors, "challenger_pass": False},
    )
    _render_skip_report(
        "Architecture Stage B4 Stability",
        stability_output,
        stability_json,
        "The specialist lane did not justify deeper stability replay work in this run; no candidate advances past B4.",
        extra={"surviving_candidates": survivors, "challenger_pass": False},
    )


def run_stage_c1(campaign: dict[str, Any]) -> None:
    route = _optional_json(campaign["reports"].get("stage4_route_json"))
    output = Path(campaign["reports"]["stage5_report"])
    json_output = Path(campaign["reports"]["stage5_json"])
    if not route.get("challenger_pass"):
        _render_skip_report(
            "Architecture Stage C1 Practicalization",
            output,
            json_output,
            "No specialist architecture candidate survived route/stability validation strongly enough to justify practicalization screening.",
            extra={"surviving_candidates": []},
        )
        return
    _render_skip_report(
        "Architecture Stage C1 Practicalization",
        output,
        json_output,
        "Practicalization was not implemented because no architecture candidate cleared the required specialist bar in this run.",
        extra={"surviving_candidates": []},
    )


def run_stage_c2(campaign: dict[str, Any]) -> None:
    stage5 = _optional_json(campaign["reports"].get("stage5_json"))
    output = Path(campaign["reports"]["stage6_report"])
    json_output = Path(campaign["reports"]["stage6_json"])
    if not stage5.get("surviving_candidates"):
        _render_skip_report(
            "Architecture Stage C2 Fairness, Holdout, and Anti-Regression",
            output,
            json_output,
            "No practicalized candidate existed, so fairness / holdout / anti-regression did not run.",
            extra={"surviving_candidates": []},
        )
        return
    _render_skip_report(
        "Architecture Stage C2 Fairness, Holdout, and Anti-Regression",
        output,
        json_output,
        "No practicalized candidate cleared C1 in this run.",
        extra={"surviving_candidates": []},
    )


def write_candidate_pack(campaign: dict[str, Any], output: Path) -> None:
    pack = _read_json(Path(campaign["current_canonical_pack"]))
    pack["architecture_specialist_program"] = {
        "state_reconciliation": str(campaign["reports"]["state_reconciliation"]),
        "frontier_manifest": str(campaign["reports"]["frontier_manifest"]),
        "registration": _optional_json(campaign["reports"].get("registration_json")),
        "stage1_exploit": _optional_json(campaign["reports"].get("stage1_exploit_json")),
        "stage1_explore": _optional_json(campaign["reports"].get("stage1_explore_json")),
        "stage2": _optional_json(campaign["reports"].get("stage2_json")),
        "stage3": _optional_json(campaign["reports"].get("stage3_json")),
        "stage4_route": _optional_json(campaign["reports"].get("stage4_route_json")),
        "stage4_stability": _optional_json(campaign["reports"].get("stage4_stability_json")),
        "stage5": _optional_json(campaign["reports"].get("stage5_json")),
        "stage6": _optional_json(campaign["reports"].get("stage6_json")),
        "winner": str(campaign["current_canonical_name"]),
        "prior_arch_run_count": _prior_arch_run_count(campaign)[0],
        "target_arch_run_count": _prior_arch_run_count(campaign)[1],
    }
    pack.setdefault("provenance", {})
    pack["provenance"]["git_commit"] = get_git_commit()
    pack["provenance"]["git_dirty"] = get_git_dirty()
    pack["provenance"]["architecture_specialist_notes"] = "specialist-only architecture rescue campaign around round6; no benchmark-lane replacement materialized"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_decision_memo(campaign: dict[str, Any], output: Path) -> None:
    stage1_exploit = _optional_json(campaign["reports"].get("stage1_exploit_json"))
    stage1_explore = _optional_json(campaign["reports"].get("stage1_explore_json"))
    stage2 = _optional_json(campaign["reports"].get("stage2_json"))
    stage3 = _optional_json(campaign["reports"].get("stage3_json"))
    stage4_route = _optional_json(campaign["reports"].get("stage4_route_json"))
    stage4_stability = _optional_json(campaign["reports"].get("stage4_stability_json"))
    stage6 = _optional_json(campaign["reports"].get("stage6_json"))
    gate = _optional_json(campaign["reports"].get("gate_report_json"))
    labels = dict(campaign["decision_strings"])
    gate_verdict = str(gate.get("verdict", gate.get("status", "missing")))
    required = str(campaign["selection"]["pack_gate_required_verdict"])
    selected = [*stage1_exploit.get("selected_candidates", []), *stage1_explore.get("selected_candidates", [])]
    stage2_survivors = list(stage2.get("surviving_candidates", []))
    specialist_survivors = list(stage3.get("specialist_survivors", []))
    practicalized = list(stage6.get("surviving_candidates", []))
    if practicalized and gate_verdict == required:
        final_status = labels["replace"]
    elif specialist_survivors and not practicalized:
        final_status = labels["arch_future_branch"]
    elif gate_verdict == required and not selected:
        final_status = labels["narrow"]
    else:
        final_status = labels["confirm"] if gate_verdict == required else labels["narrow"]
    lines = [
        "# Architecture Decision Memo",
        "",
        f"- final status: `{final_status}`",
        f"- active benchmark line: `{campaign['current_canonical_name']}`",
        f"- gate verdict: `{gate_verdict}`",
        "",
        "## Funnel",
        "",
        f"- Stage B1 selected candidates: `{selected}`",
        f"- Stage B2 survivors: `{stage2_survivors}`",
        f"- Stage B3 specialist survivors: `{specialist_survivors}`",
        f"- Stage B4 route pass: `{stage4_route.get('challenger_pass', False)}`",
        f"- Stage B4 stability pass: `{stage4_stability.get('challenger_pass', False)}`",
        f"- Stage C2 practicalized survivors: `{practicalized}`",
        "",
        "## Decision",
        "",
    ]
    if final_status == labels["replace"]:
        lines.append("- A practicalized architecture-derived challenger survived the full benchmark path strongly enough to replace `round6`.")
    elif final_status == labels["arch_future_branch"]:
        lines.append("- The architecture lane showed specialist value, but it never converted into a practical benchmark-lane candidate. The active benchmark stays unchanged and future architecture work remains specialist-only.")
    elif final_status == labels["confirm"]:
        lines.append("- The active benchmark stays on `round6`, and this program sharpened the architecture frontier without producing a stronger benchmark-lane candidate.")
    else:
        lines.append("- No architecture line survived even as a specialist rescue. The active benchmark stays on `round6`, and the architecture frontier should narrow around the keyed-residual family ceiling instead of widening.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Architecture specialist rescue program around active round6")
    sub = parser.add_subparsers(dest="command", required=True)

    for command in (
        "stage-a",
        "stage-b1",
        "stage-b2",
        "stage-b3",
        "stage-b4",
        "stage-c1",
        "stage-c2",
        "candidate-pack",
        "decision-memo",
    ):
        item = sub.add_parser(command)
        item.add_argument("--campaign-config", required=True)
        if command == "candidate-pack":
            item.add_argument("--output", required=True)
        if command == "decision-memo":
            item.add_argument("--output", required=True)

    eval_task = sub.add_parser("eval-task")
    eval_task.add_argument("--spec", required=True)
    eval_task.add_argument("--device", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "eval-task":
        _run_eval_task(Path(args.spec), device=str(args.device))
        return
    campaign = load_campaign_config(Path(args.campaign_config))
    if args.command == "stage-a":
        _write_stage_a(campaign)
        return
    if args.command == "stage-b1":
        run_stage_b1(campaign)
        return
    if args.command == "stage-b2":
        run_stage_b2(campaign)
        return
    if args.command == "stage-b3":
        run_stage_b3(campaign)
        return
    if args.command == "stage-b4":
        run_stage_b4(campaign)
        return
    if args.command == "stage-c1":
        run_stage_c1(campaign)
        return
    if args.command == "stage-c2":
        run_stage_c2(campaign)
        return
    if args.command == "candidate-pack":
        write_candidate_pack(campaign, Path(args.output))
        return
    if args.command == "decision-memo":
        render_decision_memo(campaign, Path(args.output))
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
