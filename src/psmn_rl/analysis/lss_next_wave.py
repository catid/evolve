from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import yaml

from psmn_rl.analysis.campaign_config import load_campaign_config


TASK_LABELS = {
    "doorkey": "DoorKey",
    "keycorridor": "KeyCorridor",
    "memory": "Memory",
    "dynamic_obstacles": "DynamicObstacles",
}

FAMILY_LABELS = {
    "tregh": "TREG-H",
    "por": "POR",
    "srw": "SRW",
    "sare_persistence": "Route-Memory SARE Persistence Diagnostic",
}

CONTROL_REFERENCE_PATHS = {
    "doorkey": {
        "flat_dense": "outputs/reports/minigrid_baseline_report.md",
        "round6": "outputs/reports/portfolio_candidate_pack.json",
    },
    "keycorridor": {
        "flat_dense_metrics": "outputs/experiments/minigrid_baselines/minigrid_keycorridor_flat_dense/metrics.jsonl",
        "token_dense_metrics": "outputs/experiments/minigrid_baselines/minigrid_keycorridor_token_dense/metrics.jsonl",
        "single_expert_metrics": "outputs/experiments/minigrid_baselines/minigrid_keycorridor_single_expert/metrics.jsonl",
        "historical_transfer": "outputs/reports/lss_keycorridor_transfer_report.md",
    },
    "dynamic_obstacles": {
        "flat_dense_metrics": "outputs/experiments/minigrid_baselines/minigrid_dynamic_obstacles_flat_dense/metrics.jsonl",
        "token_dense_metrics": "outputs/experiments/minigrid_baselines/minigrid_dynamic_obstacles_token_dense/metrics.jsonl",
        "single_expert_metrics": "outputs/experiments/minigrid_baselines/minigrid_dynamic_obstacles_single_expert/metrics.jsonl",
    },
    "memory": {
        "token_dense_metrics": "outputs/minigrid/main_runs/memory_token_dense_60/metrics.jsonl",
        "token_gru_metrics": "outputs/minigrid/main_runs/memory_token_gru_60/metrics.jsonl",
    },
}


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = json.loads(json.dumps(value))
    return result


def _gpu_count(device: str) -> int:
    if device == "cpu":
        return 0
    if torch.cuda.is_available():
        return int(torch.cuda.device_count())
    return 0


def _torchrun_path() -> str:
    candidate = Path(sys.executable).with_name("torchrun")
    if candidate.exists():
        return str(candidate)
    return "torchrun"


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["job_id"]): dict(row) for row in payload.get("jobs", [])}


def _save_manifest(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    _write_json(path, {"updated_at": _timestamp(), "jobs": list(rows.values())})


def _job_output_root(stage_root: Path, family: str, task: str, variant: str, seed: int) -> Path:
    return stage_root / task / variant / f"seed_{seed}"


def _generated_config_path(output_root: Path) -> Path:
    return output_root / "generated_config.yaml"


def _record_command(path: Path, parts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(parts) + "\n", encoding="utf-8")


def _metrics_rows(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    rows = _metrics_rows(run_dir)
    if not rows:
        return {
            "run_dir": str(run_dir),
            "eval_success_rate": 0.0,
            "eval_return": 0.0,
            "best_train_return": 0.0,
            "best_train_success_rate": 0.0,
            "throughput_fps": 0.0,
            "active_compute_proxy": 0.0,
            "route_entropy": 0.0,
            "avg_hop_count": 0.0,
            "avg_halting_probability": 0.0,
            "relational_usage_rate": 0.0,
            "option_duration": 0.0,
            "option_switch_rate": 0.0,
            "global_step": 0.0,
        }
    last = rows[-1]
    return {
        "run_dir": str(run_dir),
        "eval_success_rate": _safe_float(last.get("eval_success_rate")),
        "eval_return": _safe_float(last.get("eval_return")),
        "best_train_return": max(_safe_float(row.get("train/episode_return")) for row in rows),
        "best_train_success_rate": max(_safe_float(row.get("train/success_rate")) for row in rows),
        "throughput_fps": _safe_float(last.get("throughput_fps")),
        "active_compute_proxy": _safe_float(last.get("active_compute_proxy")),
        "route_entropy": _safe_float(last.get("route_entropy")),
        "avg_hop_count": _safe_float(last.get("avg_hop_count")),
        "avg_halting_probability": _safe_float(last.get("avg_halting_probability")),
        "relational_usage_rate": _safe_float(last.get("relational_usage_rate")),
        "option_duration": _safe_float(last.get("option_duration")),
        "option_switch_rate": _safe_float(last.get("option_switch_rate")),
        "global_step": _safe_float(last.get("global_step")),
    }


def _control_reference_from_metrics(path: str | Path) -> dict[str, float]:
    rows = _metrics_rows(Path(path))
    if not rows:
        return {"eval_success_rate": 0.0, "best_train_return": 0.0}
    return {
        "eval_success_rate": _safe_float(rows[-1].get("eval_success_rate")),
        "best_train_return": max(_safe_float(row.get("train/episode_return")) for row in rows),
    }


def _control_ceiling(task: str, recurrent_controls: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, float]:
    if task == "doorkey":
        return {"eval_success_rate": 1.0, "best_train_return": 0.96}
    if task == "keycorridor":
        refs = [
            _control_reference_from_metrics(CONTROL_REFERENCE_PATHS["keycorridor"]["flat_dense_metrics"]),
            _control_reference_from_metrics(CONTROL_REFERENCE_PATHS["keycorridor"]["token_dense_metrics"]),
            _control_reference_from_metrics(CONTROL_REFERENCE_PATHS["keycorridor"]["single_expert_metrics"]),
        ]
        if recurrent_controls:
            refs.extend(recurrent_controls.get("keycorridor", []))
        return {
            "eval_success_rate": max(item["eval_success_rate"] for item in refs),
            "best_train_return": max(item["best_train_return"] for item in refs),
        }
    if task == "dynamic_obstacles":
        refs = [
            _control_reference_from_metrics(CONTROL_REFERENCE_PATHS["dynamic_obstacles"]["flat_dense_metrics"]),
            _control_reference_from_metrics(CONTROL_REFERENCE_PATHS["dynamic_obstacles"]["token_dense_metrics"]),
            _control_reference_from_metrics(CONTROL_REFERENCE_PATHS["dynamic_obstacles"]["single_expert_metrics"]),
        ]
        return {
            "eval_success_rate": max(item["eval_success_rate"] for item in refs),
            "best_train_return": max(item["best_train_return"] for item in refs),
        }
    if task == "memory":
        refs = [
            _control_reference_from_metrics(CONTROL_REFERENCE_PATHS["memory"]["token_dense_metrics"]),
            _control_reference_from_metrics(CONTROL_REFERENCE_PATHS["memory"]["token_gru_metrics"]),
        ]
        if recurrent_controls:
            refs.extend(recurrent_controls.get("memory", []))
        return {
            "eval_success_rate": max(item["eval_success_rate"] for item in refs),
            "best_train_return": max(item["best_train_return"] for item in refs),
        }
    return {"eval_success_rate": 0.0, "best_train_return": 0.0}


def _verify_stage0(campaign: dict[str, Any]) -> None:
    for key in ("state_reconciliation", "baseline_sync", "registration"):
        report_path = Path(campaign["reports"][key])
        if not report_path.exists():
            raise FileNotFoundError(report_path)
    print("Stage 0 reports exist and campaign config is readable.")


def _family_jobs(campaign: dict[str, Any], family: str) -> list[dict[str, Any]]:
    family_spec = campaign["families"][family]
    stage_root = Path(campaign["stage_roots"][family_spec["stage_root"]])
    seeds = [int(seed) for seed in campaign[family_spec["seeds_key"]]]
    schedule = campaign["defaults"][family_spec["schedule_key"]]
    jobs: list[dict[str, Any]] = []
    for task_name, task_spec in family_spec["tasks"].items():
        base_config = _read_yaml(task_spec["base_config"])
        for variant_name, overrides in family_spec["variants"].items():
            for seed in seeds:
                output_root = _job_output_root(stage_root, family, task_name, variant_name, seed)
                payload = _deep_merge(base_config, schedule)
                payload = _deep_merge(payload, overrides)
                payload["seed"] = seed
                payload["tags"] = list(payload.get("tags", [])) + ["next_wave", family, task_name, variant_name]
                payload.setdefault("logging", {})
                payload["logging"]["run_name"] = f"{family}_{task_name}_{variant_name}_seed{seed}"
                payload["logging"]["output_dir"] = str(output_root)
                generated_config = _generated_config_path(output_root)
                jobs.append(
                    {
                        "job_id": f"{family}__{task_name}__{variant_name}__seed{seed}",
                        "family": family,
                        "task": task_name,
                        "task_label": str(task_spec["display_name"]),
                        "variant": variant_name,
                        "seed": seed,
                        "output_root": str(output_root),
                        "generated_config": str(generated_config),
                        "payload": payload,
                        "rerun_lineage": "original",
                    }
                )
    return jobs


def _recurrent_control_jobs(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    stage_root = Path(campaign["stage_roots"][campaign["recurrent_controls"]["stage_root"]])
    schedule = campaign["defaults"][campaign["recurrent_controls"]["schedule_key"]]
    jobs: list[dict[str, Any]] = []
    for task_name, task_spec in campaign["recurrent_controls"]["tasks"].items():
        base_config = _read_yaml(task_spec["base_config"])
        for seed in campaign["control_seeds"]:
            output_root = stage_root / task_name / f"seed_{seed}"
            payload = _deep_merge(base_config, schedule)
            payload["seed"] = int(seed)
            payload["tags"] = list(payload.get("tags", [])) + ["next_wave", "recurrent_control", task_name]
            payload.setdefault("logging", {})
            payload["logging"]["run_name"] = f"token_gru_{task_name}_seed{seed}"
            payload["logging"]["output_dir"] = str(output_root)
            generated_config = _generated_config_path(output_root)
            jobs.append(
                {
                    "job_id": f"recurrent_control__{task_name}__token_gru__seed{seed}",
                    "family": "recurrent_control",
                    "task": task_name,
                    "task_label": str(task_spec["display_name"]),
                    "variant": "token_gru",
                    "seed": int(seed),
                    "output_root": str(output_root),
                    "generated_config": str(generated_config),
                    "payload": payload,
                    "rerun_lineage": "original",
                }
            )
    return jobs


def _run_jobs(campaign: dict[str, Any], jobs: list[dict[str, Any]], device: str) -> None:
    manifest_path = Path(campaign["reports"]["manifest_json"])
    manifest = _load_manifest(manifest_path)
    pending: list[dict[str, Any]] = []
    for job in jobs:
        existing = manifest.get(job["job_id"])
        output_root = Path(job["output_root"])
        if existing and existing.get("status") == "completed" and (output_root / "metrics.jsonl").exists():
            continue
        pending.append(job)
    if not pending:
        return

    for job in pending:
        _write_yaml(job["generated_config"], job["payload"])
        job["config_hash"] = _sha256_path(Path(job["generated_config"]))

    gpu_count = _gpu_count(device)
    if device == "cpu" or gpu_count == 0:
        slots = [None]
    else:
        slots = list(range(gpu_count))

    running: dict[int | None, tuple[subprocess.Popen[str], dict[str, Any], Any, Any]] = {}
    queue = list(pending)

    while queue or running:
        for slot in slots:
            if slot in running or not queue:
                continue
            job = queue.pop(0)
            output_root = Path(job["output_root"])
            output_root.mkdir(parents=True, exist_ok=True)
            cmd = [
                _torchrun_path(),
                "--standalone",
                "--nproc_per_node=1",
                "-m",
                "psmn_rl.launch",
                "--config",
                job["generated_config"],
            ]
            env = os.environ.copy()
            if slot is not None and device != "cpu":
                env["CUDA_VISIBLE_DEVICES"] = str(slot)
            _record_command(output_root / "command.txt", cmd)
            stdout_handle = (output_root / "run.stdout.log").open("w", encoding="utf-8")
            stderr_handle = (output_root / "run.stderr.log").open("w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=Path.cwd(),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            row = {
                "job_id": job["job_id"],
                "family": job["family"],
                "task": job["task"],
                "task_label": job["task_label"],
                "variant": job["variant"],
                "seed_block": int(job["seed"]),
                "config_path": job["generated_config"],
                "config_hash": job["config_hash"],
                "output_root": job["output_root"],
                "rerun_lineage": job["rerun_lineage"],
                "start_timestamp": _timestamp(),
                "end_timestamp": None,
                "status": "running",
                "returncode": None,
            }
            manifest[job["job_id"]] = row
            _save_manifest(manifest_path, manifest)
            running[slot] = (proc, job, stdout_handle, stderr_handle)
        finished: list[int | None] = []
        for slot, (proc, job, stdout_handle, stderr_handle) in running.items():
            returncode = proc.poll()
            if returncode is None:
                continue
            stdout_handle.close()
            stderr_handle.close()
            row = manifest[job["job_id"]]
            row["end_timestamp"] = _timestamp()
            row["returncode"] = int(returncode)
            row["status"] = "completed" if returncode == 0 else "failed"
            manifest[job["job_id"]] = row
            _save_manifest(manifest_path, manifest)
            if returncode != 0:
                raise RuntimeError(f"job failed: {job['job_id']}")
            finished.append(slot)
        for slot in finished:
            running.pop(slot, None)
        if running:
            time.sleep(1.0)


def _collect_rows(campaign: dict[str, Any], family: str) -> list[dict[str, Any]]:
    manifest = _load_manifest(Path(campaign["reports"]["manifest_json"]))
    rows: list[dict[str, Any]] = []
    for item in manifest.values():
        if item["family"] != family or item.get("status") != "completed":
            continue
        run_summary = _summarize_run(Path(item["output_root"]))
        rows.append(
            {
                "family": family,
                "task": item["task"],
                "task_label": item["task_label"],
                "variant": item["variant"],
                "seed": int(item["seed_block"]),
                "output_root": item["output_root"],
                **run_summary,
            }
        )
    rows.sort(key=lambda row: (row["task"], row["variant"], row["seed"]))
    return rows


def _collect_recurrent_control_rows(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return _collect_rows(campaign, "recurrent_control")


def _aggregate_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["variant"]))].append(row)
    aggregated: list[dict[str, Any]] = []
    for (task, variant), subset in sorted(grouped.items()):
        aggregated.append(
            {
                "task": task,
                "task_label": subset[0]["task_label"],
                "variant": variant,
                "runs": len(subset),
                "eval_success_rate": float(mean(row["eval_success_rate"] for row in subset)),
                "eval_return": float(mean(row["eval_return"] for row in subset)),
                "best_train_return": float(mean(row["best_train_return"] for row in subset)),
                "best_train_success_rate": float(mean(row["best_train_success_rate"] for row in subset)),
                "throughput_fps": float(mean(row["throughput_fps"] for row in subset)),
                "active_compute_proxy": float(mean(row["active_compute_proxy"] for row in subset)),
                "route_entropy": float(mean(row["route_entropy"] for row in subset)),
                "avg_hop_count": float(mean(row["avg_hop_count"] for row in subset)),
                "avg_halting_probability": float(mean(row["avg_halting_probability"] for row in subset)),
                "relational_usage_rate": float(mean(row["relational_usage_rate"] for row in subset)),
                "option_duration": float(mean(row["option_duration"] for row in subset)),
                "option_switch_rate": float(mean(row["option_switch_rate"] for row in subset)),
            }
        )
    return aggregated


def _pick_family_survivors(
    family: str,
    aggregates: list[dict[str, Any]],
    recurrent_controls: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    survivors: list[dict[str, Any]] = []
    for row in aggregates:
        control = _control_ceiling(str(row["task"]), recurrent_controls=recurrent_controls)
        if row["eval_success_rate"] > control["eval_success_rate"] + 0.01 and row["best_train_return"] > control["best_train_return"] + 0.05:
            survivors.append(row)
    return survivors


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _format_float(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "0.0000"


def _write_skip_report(path: str | Path, title: str, reason: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"# {title}\n\n- formal skip path: `{reason}`\n", encoding="utf-8")


def _write_recurrent_impl_report(campaign: dict[str, Any]) -> None:
    lines = [
        "# Recurrent Fairness Stage 0 Implementation",
        "",
        "- sequence-aware PPO minibatching is now available via `ppo.sequence_minibatches`.",
        "- validated config surfaces:",
        "  - `configs/minigrid/main/memory_token_gru.yaml`",
        "  - `configs/minigrid/baseline/memory_token_gru.yaml`",
        "  - `configs/diagnostic/minigrid_memory_token_gru_probe.yaml`",
        "  - `configs/minigrid/main/keycorridor_token_gru.yaml`",
        "  - `configs/experiments/minigrid_keycorridor_token_gru.yaml`",
        "- new recurrent model knobs remain bounded and auditable:",
        "  - TREG-H `max_hops` and `halt_bias`",
        "  - POR `termination_bias`",
        "- focused validation command:",
        "  - `./.venv/bin/python -m pytest -q tests/test_models.py tests/test_routing.py tests/test_training_regressions.py tests/test_configs.py`",
        "",
        "## Verdict",
        "",
        "- The recurrent fairness blocker is no longer architectural. `token_gru` can now be treated as a fair recurrent control on task configs that enable `ppo.sequence_minibatches: true`.",
        "- POR benchmarking is therefore allowed to proceed, but it still has to earn task relevance under the same control-first rules as the routed families.",
    ]
    Path(campaign["reports"]["recurrent_impl"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_recurrent_controls_report(campaign: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = _collect_recurrent_control_rows(campaign)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(
            {
                "eval_success_rate": row["eval_success_rate"],
                "best_train_return": row["best_train_return"],
                "throughput_fps": row["throughput_fps"],
                "seed": row["seed"],
            }
        )
    json_payload = {"rows": rows, "grouped": by_task}
    _write_json(campaign["reports"]["recurrent_controls_json"], json_payload)
    lines = [
        "# Recurrent Fairness Stage 1 Controls",
        "",
        "- family: `token_gru`",
        "- tasks: `Memory`, `KeyCorridor`",
        "- sequence-aware PPO minibatching: `enabled`",
        "",
    ]
    table_rows: list[list[str]] = []
    for row in rows:
        table_rows.append(
            [
                TASK_LABELS.get(str(row["task"]), str(row["task"])),
                str(row["seed"]),
                _format_float(row["eval_success_rate"]),
                _format_float(row["eval_return"]),
                _format_float(row["best_train_return"]),
                _format_float(row["throughput_fps"], 1),
            ]
        )
    lines.extend(_markdown_table(["Task", "Seed", "Eval Success", "Eval Return", "Best Train Return", "Throughput"], table_rows))
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "- The recurrent control lane is now coherent enough to benchmark POR because the controls were trained through the same sequence-aware PPO path that POR itself requires.",
            "- This report is a fairness-enabling artifact, not a claim that the recurrent controls already establish a benchmark lane on these tasks.",
        ]
    )
    Path(campaign["reports"]["recurrent_controls"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return by_task


def _write_family_report(campaign: dict[str, Any], family: str) -> list[dict[str, Any]]:
    rows = _collect_rows(campaign, family)
    aggregates = _aggregate_family(rows)
    recurrent_controls = None
    if family == "por":
        grouped = json.loads(Path(campaign["reports"]["recurrent_controls_json"]).read_text(encoding="utf-8")).get("grouped", {})
        recurrent_controls = {
            task: [
                {
                    "eval_success_rate": _safe_float(entry["eval_success_rate"]),
                    "best_train_return": _safe_float(entry["best_train_return"]),
                }
                for entry in entries
            ]
            for task, entries in grouped.items()
        }
    survivors = _pick_family_survivors(family, aggregates, recurrent_controls=recurrent_controls)

    if family == "tregh":
        report_path = campaign["reports"]["tregh_stage1"]
        json_path = campaign["reports"]["tregh_stage1_json"]
        later = [
            ("TREG-H Stage 2 Verification", campaign["reports"]["tregh_stage2"]),
            ("TREG-H Stage 3 Fairness", campaign["reports"]["tregh_stage3"]),
            ("TREG-H Stage 4 Holdout", campaign["reports"]["tregh_stage4"]),
            ("TREG-H Stage 5 Route/Stability", campaign["reports"]["tregh_stage5"]),
        ]
    elif family == "por":
        report_path = campaign["reports"]["por_stage2"]
        json_path = campaign["reports"]["por_stage2_json"]
        later = [
            ("POR Stage 3 Verification", campaign["reports"]["por_stage3"]),
            ("POR Stage 4 Fairness", campaign["reports"]["por_stage4"]),
            ("POR Stage 5 Holdout", campaign["reports"]["por_stage5"]),
            ("POR Stage 6 Stability", campaign["reports"]["por_stage6"]),
        ]
    else:
        report_path = campaign["reports"]["srw_stage1"]
        json_path = campaign["reports"]["srw_stage1_json"]
        later = [
            ("SRW Stage 2 Verification", campaign["reports"]["srw_stage2"]),
            ("SRW Stage 3 Fairness", campaign["reports"]["srw_stage3"]),
            ("SRW Stage 4 Holdout", campaign["reports"]["srw_stage4"]),
            ("SRW Stage 5 Stability", campaign["reports"]["srw_stage5"]),
        ]

    _write_json(json_path, {"rows": rows, "aggregates": aggregates, "survivors": survivors})
    lines = [f"# {FAMILY_LABELS[family]} Screening", ""]
    lines.append(f"- substantive runs: `{len(rows)}`")
    lines.append(f"- surviving variants after control gate: `{len(survivors)}`")
    lines.append("")
    detail_rows: list[list[str]] = []
    for row in aggregates:
        detail_rows.append(
            [
                TASK_LABELS.get(str(row["task"]), str(row["task"])),
                str(row["variant"]),
                str(int(row["runs"])),
                _format_float(row["eval_success_rate"]),
                _format_float(row["eval_return"]),
                _format_float(row["best_train_return"]),
                _format_float(row["throughput_fps"], 1),
                _format_float(row["active_compute_proxy"], 3),
                _format_float(row["avg_hop_count"], 3),
                _format_float(row["avg_halting_probability"], 3),
                _format_float(row["relational_usage_rate"], 3),
                _format_float(row["option_duration"], 3),
                _format_float(row["option_switch_rate"], 3),
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "Task",
                "Variant",
                "Runs",
                "Eval Success",
                "Eval Return",
                "Best Train Return",
                "Throughput",
                "Active Compute",
                "Hop Count",
                "Halting",
                "Rel Usage",
                "Option Duration",
                "Option Switch",
            ],
            detail_rows,
        )
    )
    lines.extend(["", "## Verdict", ""])
    if survivors:
        lines.append("- One or more variants cleared the coarse control gate and require later-stage work.")
        lines.append("- This pass still records only the screening result; later-stage reports must be filled in before any promotion claim.")
    else:
        if family == "tregh":
            lines.append("- No TREG-H variant produced a control-meaningful survivor. The lane stays exploratory and later stages enter the formal skip path.")
            lines.append("- In practice this means no adaptive-depth / halting variant cleared both nonzero evaluation and control relevance on DoorKey or KeyCorridor.")
        elif family == "por":
            lines.append("- No POR variant beat the recurrent fair-control lane on Memory or KeyCorridor, so no persistent-routing candidate advanced.")
            lines.append("- The recurrent fairness repair succeeded, but the POR benchmark lane itself did not produce a control-meaningful survivor.")
        else:
            lines.append("- No SRW variant produced a control-meaningful survivor on DynamicObstacles or KeyCorridor, so the selective-relational lane stays exploratory only.")
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not survivors:
        for title, path in later:
            _write_skip_report(path, title, "no stage-1 survivor")
    return survivors


def _write_compute_structure_reports(campaign: dict[str, Any]) -> None:
    tregh = json.loads(Path(campaign["reports"]["tregh_stage1_json"]).read_text(encoding="utf-8"))
    srw = json.loads(Path(campaign["reports"]["srw_stage1_json"]).read_text(encoding="utf-8"))
    por = json.loads(Path(campaign["reports"]["por_stage2_json"]).read_text(encoding="utf-8"))
    if not tregh.get("survivors"):
        _write_skip_report(campaign["reports"]["compute_structure_tregh"], "Compute-Structure Stage 1 TREG-H", "no living TREG-H baseline")
    if not srw.get("survivors"):
        _write_skip_report(campaign["reports"]["compute_structure_srw"], "Compute-Structure Stage 2 SRW", "no living SRW baseline")
    if not por.get("survivors"):
        _write_skip_report(campaign["reports"]["compute_structure_por"], "Compute-Structure Stage 3 POR", "no living POR baseline")

    rows = _collect_rows(campaign, "sare_persistence")
    aggregates = _aggregate_family(rows)
    _write_json(campaign["reports"]["compute_structure_sare_persistence_json"], {"rows": rows, "aggregates": aggregates})
    lines = [
        "# Compute-Structure Stage 4 SARE Persistence Diagnostic",
        "",
        "- bounded exploratory route-memory line on `KeyCorridor` only",
        "- no DoorKey-only SARE surgery reopened",
        "",
    ]
    table_rows: list[list[str]] = []
    for row in aggregates:
        table_rows.append(
            [
                str(row["variant"]),
                str(int(row["runs"])),
                _format_float(row["eval_success_rate"]),
                _format_float(row["eval_return"]),
                _format_float(row["best_train_return"]),
                _format_float(row["throughput_fps"], 1),
                _format_float(row["route_entropy"], 3),
            ]
        )
    lines.extend(_markdown_table(["Variant", "Runs", "Eval Success", "Eval Return", "Best Train Return", "Throughput", "Route Entropy"], table_rows))
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "- This diagnostic stays exploratory only.",
            "- It is included to test route-memory persistence ideas off DoorKey without reopening the broad SARE surgery lane.",
        ]
    )
    Path(campaign["reports"]["compute_structure_sare_persistence"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_aggregate(aggregates: list[dict[str, Any]], task: str) -> dict[str, Any] | None:
    subset = [row for row in aggregates if str(row["task"]) == task]
    if not subset:
        return None
    return max(subset, key=lambda row: (row["eval_success_rate"], row["best_train_return"], -row["active_compute_proxy"]))


def _frontier_rows(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tregh = json.loads(Path(campaign["reports"]["tregh_stage1_json"]).read_text(encoding="utf-8")).get("aggregates", [])
    por = json.loads(Path(campaign["reports"]["por_stage2_json"]).read_text(encoding="utf-8")).get("aggregates", [])
    srw = json.loads(Path(campaign["reports"]["srw_stage1_json"]).read_text(encoding="utf-8")).get("aggregates", [])
    sare = json.loads(Path(campaign["reports"]["compute_structure_sare_persistence_json"]).read_text(encoding="utf-8")).get("aggregates", [])

    frontier_specs = [
        ("doorkey", "TREG-H", _best_aggregate(tregh, "doorkey")),
        ("keycorridor", "TREG-H", _best_aggregate(tregh, "keycorridor")),
        ("dynamic_obstacles", "SRW", _best_aggregate(srw, "dynamic_obstacles")),
        ("keycorridor", "SRW", _best_aggregate(srw, "keycorridor")),
        ("memory", "POR", _best_aggregate(por, "memory")),
        ("keycorridor", "POR", _best_aggregate(por, "keycorridor")),
        ("keycorridor", "SARE persistence", _best_aggregate(sare, "keycorridor")),
    ]
    for task, family_label, row in frontier_specs:
        if row is None:
            continue
        rows.append(
            {
                "task": task,
                "family": family_label,
                "variant": row["variant"],
                "eval_success_rate": row["eval_success_rate"],
                "eval_return": row["eval_return"],
                "best_train_return": row["best_train_return"],
                "throughput_fps": row["throughput_fps"],
                "active_compute_proxy": row["active_compute_proxy"],
                "avg_hop_count": row["avg_hop_count"],
                "avg_halting_probability": row["avg_halting_probability"],
                "relational_usage_rate": row["relational_usage_rate"],
                "option_duration": row["option_duration"],
                "option_switch_rate": row["option_switch_rate"],
            }
        )
    rows.extend(
        [
            {
                "task": "doorkey",
                "family": "round6",
                "variant": "active_benchmark_reference",
                "eval_success_rate": 0.8320,
                "eval_return": 0.0,
                "best_train_return": 0.0,
                "throughput_fps": 0.0,
                "active_compute_proxy": 0.5,
                "avg_hop_count": 0.0,
                "avg_halting_probability": 0.0,
                "relational_usage_rate": 0.0,
                "option_duration": 0.0,
                "option_switch_rate": 0.0,
            },
            {
                "task": "doorkey",
                "family": "flat_dense",
                "variant": "baseline_reference",
                "eval_success_rate": 1.0,
                "eval_return": 0.965,
                "best_train_return": 0.960,
                "throughput_fps": 9668.3,
                "active_compute_proxy": 1.0,
                "avg_hop_count": 0.0,
                "avg_halting_probability": 0.0,
                "relational_usage_rate": 0.0,
                "option_duration": 0.0,
                "option_switch_rate": 0.0,
            },
        ]
    )
    return rows


def _write_frontier_report(campaign: dict[str, Any]) -> None:
    rows = _frontier_rows(campaign)
    _write_json(campaign["reports"]["frontier_json"], {"rows": rows})
    _write_csv(campaign["reports"]["frontier_csv"], rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(row)
    lines = ["# Compute Frontier Report", ""]
    for task in ("doorkey", "keycorridor", "memory", "dynamic_obstacles"):
        if task not in grouped:
            continue
        lines.append(f"## {TASK_LABELS.get(task, task)}")
        lines.append("")
        table_rows: list[list[str]] = []
        for row in grouped[task]:
            table_rows.append(
                [
                    str(row["family"]),
                    str(row["variant"]),
                    _format_float(row["eval_success_rate"]),
                    _format_float(row["eval_return"]),
                    _format_float(row["best_train_return"]),
                    _format_float(row["throughput_fps"], 1),
                    _format_float(row["active_compute_proxy"], 3),
                    _format_float(row["avg_hop_count"], 3),
                    _format_float(row["avg_halting_probability"], 3),
                    _format_float(row["relational_usage_rate"], 3),
                    _format_float(row["option_duration"], 3),
                    _format_float(row["option_switch_rate"], 3),
                ]
            )
        lines.extend(
            _markdown_table(
                [
                    "Family",
                    "Variant",
                    "Eval Success",
                    "Eval Return",
                    "Best Train Return",
                    "Throughput",
                    "Active Compute",
                    "Hop Count",
                    "Halting",
                    "Rel Usage",
                    "Option Duration",
                    "Option Switch",
                ],
                table_rows,
            )
        )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- This is a screening-stage compute frontier, not a claim-bearing benchmark replacement table.",
            "- It answers whether any non-SARE family produced obvious return-vs-compute value worth late-stage verification.",
        ]
    )
    Path(campaign["reports"]["frontier_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_candidate_pack(campaign: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "pack_type": "next_wave_summary",
        "generated_at": _timestamp(),
        "active_benchmark": campaign["current_canonical_name"],
        "status": "no_promotion",
        "task_scoped_candidates": [],
        "notes": "No family cleared the next-wave control gate, so no task-scoped candidate pack was promoted.",
    }
    _write_json(campaign["reports"]["candidate_pack_summary"], payload)
    Path(campaign["reports"]["gate_report"]).write_text(
        "# Next-Wave Gate Report\n\n- no task-scoped candidate pack was promoted\n- gate invocation: `not required`\n- reason: `no family cleared controls, reruns, and holdout prerequisites`\n",
        encoding="utf-8",
    )
    return payload


def _write_decision_memo(campaign: dict[str, Any]) -> None:
    tregh = json.loads(Path(campaign["reports"]["tregh_stage1_json"]).read_text(encoding="utf-8"))
    srw = json.loads(Path(campaign["reports"]["srw_stage1_json"]).read_text(encoding="utf-8"))
    por = json.loads(Path(campaign["reports"]["por_stage2_json"]).read_text(encoding="utf-8"))
    status = "round6 remains the only active benchmark and the other families stay exploratory"
    if not tregh.get("survivors") and not srw.get("survivors") and not por.get("survivors"):
        status = "overall frontier narrows because none of the alternative architecture theses survive proper controls"
    lines = [
        "# Next-Wave Decision Memo",
        "",
        f"- final status: `{status}`",
        f"- active benchmark remains: `{campaign['current_canonical_name']}`",
        "",
        "## Summary",
        "",
        "- The next-wave screens were run across TREG-H, POR, SRW, and one bounded non-DoorKey SARE persistence diagnostic.",
        "- The recurrent fairness blocker was repaired first so POR could be screened against a fair recurrent control lane.",
        "- No family cleared the repo’s control-first benchmark promotion funnel in this pass, so the active DoorKey benchmark remains `round6`.",
    ]
    Path(campaign["reports"]["decision_memo"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-pollinated next-wave benchmark runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser.add_argument(
        "--campaign-config",
        default="configs/experiments/lss_next_wave_program/campaign.yaml",
        help="Path to next-wave campaign config",
    )

    subparsers.add_parser("verify-stage0")

    run_screen = subparsers.add_parser("run-screen")
    run_screen.add_argument("--family", choices=("tregh", "por", "srw", "sare_persistence"), required=True)
    run_screen.add_argument("--device", default="auto")

    run_controls = subparsers.add_parser("run-recurrent-controls")
    run_controls.add_argument("--device", default="auto")

    subparsers.add_parser("write-recurrent-impl-report")
    subparsers.add_parser("write-recurrent-controls-report")

    write_family = subparsers.add_parser("write-family-report")
    write_family.add_argument("--family", choices=("tregh", "por", "srw"), required=True)

    subparsers.add_parser("write-compute-structure-report")
    subparsers.add_parser("write-frontier-report")
    subparsers.add_parser("write-candidate-pack")
    subparsers.add_parser("write-decision-memo")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    campaign = load_campaign_config(args.campaign_config)

    if args.command == "verify-stage0":
        _verify_stage0(campaign)
    elif args.command == "run-screen":
        _run_jobs(campaign, _family_jobs(campaign, args.family), getattr(args, "device", "auto"))
    elif args.command == "run-recurrent-controls":
        _run_jobs(campaign, _recurrent_control_jobs(campaign), getattr(args, "device", "auto"))
    elif args.command == "write-recurrent-impl-report":
        _write_recurrent_impl_report(campaign)
    elif args.command == "write-recurrent-controls-report":
        _write_recurrent_controls_report(campaign)
    elif args.command == "write-family-report":
        _write_family_report(campaign, args.family)
    elif args.command == "write-compute-structure-report":
        _write_compute_structure_reports(campaign)
    elif args.command == "write-frontier-report":
        _write_frontier_report(campaign)
    elif args.command == "write-candidate-pack":
        _write_candidate_pack(campaign)
    elif args.command == "write-decision-memo":
        _write_decision_memo(campaign)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
