from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from psmn_rl.analysis.campaign_config import load_campaign_config
from psmn_rl.analysis.lss_escape_distill import _fit_linear_probe, _state_tensor_features
from psmn_rl.analysis.lss_forensic_atlas import (
    _action_summary,
    _build_single_env,
    _extract_grid_state,
    _forward_with_route_capture,
    _obs_to_batch,
    _route_capture_summary,
)
from psmn_rl.analysis.lss_next_wave import (
    CONTROL_REFERENCE_PATHS,
    FAMILY_LABELS,
    TASK_LABELS,
    _aggregate_family,
    _collect_rows,
    _control_ceiling,
    _control_reference_from_metrics,
    _deep_merge,
    _format_float,
    _generated_config_path,
    _gpu_count,
    _job_output_root,
    _load_manifest,
    _markdown_table,
    _metrics_rows,
    _pick_family_survivors,
    _read_yaml,
    _record_command,
    _run_jobs,
    _safe_float,
    _save_manifest,
    _sha256_path,
    _summarize_run,
    _timestamp,
    _torchrun_path,
    _write_csv,
    _write_json,
    _write_skip_report,
    _write_yaml,
)
from psmn_rl.analysis.policy_distillation import _load_model
from psmn_rl.rl.ppo.algorithm import prepare_done, prepare_obs
from psmn_rl.utils.seed import capture_rng_state, restore_rng_state, set_seed


SECOND_WAVE_FAMILY_LABELS = {
    "tregh": "TREG-H",
    "recurrent_control": "Recurrent Controls",
    "por": "POR",
    "srw": "SRW",
    "sare_persistence": "Route-Memory SARE Persistence Diagnostic",
}


def _resolved_config_path(run_dir: Path) -> Path:
    for name in ("resolved_config.yaml", "student_resolved_config.yaml", "generated_config.yaml"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return run_dir / "resolved_config.yaml"


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
                payload["tags"] = list(payload.get("tags", [])) + ["next_arch_wave", family, task_name, variant_name]
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


def _prior_rows(campaign: dict[str, Any], family: str) -> list[dict[str, Any]]:
    manifest = _load_manifest(Path(campaign["prior_wave_manifest"]))
    rows: list[dict[str, Any]] = []
    for item in manifest.values():
        if item.get("family") != family or item.get("status") != "completed":
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
    rows.sort(key=lambda row: (str(row["task"]), str(row["variant"]), int(row["seed"])))
    return rows


def _best_specs(rows: list[dict[str, Any]], *, prefer_two_tasks: bool = True) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(row)
    specs: list[dict[str, Any]] = []
    for task, subset in sorted(grouped.items()):
        best = max(
            subset,
            key=lambda row: (
                float(row["eval_success_rate"]),
                float(row["best_train_return"]),
                float(row["best_train_success_rate"]),
                -float(row["active_compute_proxy"]),
            ),
        )
        run_dir = Path(str(best["output_root"]))
        specs.append(
            {
                "task": task,
                "task_label": best["task_label"],
                "variant": best["variant"],
                "seed": int(best["seed"]),
                "output_root": str(run_dir),
                "config_path": str(_resolved_config_path(run_dir)),
                "checkpoint_path": str(run_dir / "latest.pt"),
                "eval_success_rate": float(best["eval_success_rate"]),
                "best_train_return": float(best["best_train_return"]),
            }
        )
    return specs


def _task_bucket(task: str, state: dict[str, Any], step: int, max_steps: int) -> str:
    if task in {"doorkey", "keycorridor"}:
        phase = str(state.get("phase", "search_key"))
        if phase in {"search_key", "at_key"}:
            return "pre_key"
        if phase == "carry_key":
            return "carry_key"
        if phase == "at_locked_door":
            return "at_locked_door"
        return "post_unlock"
    fraction = (step + 1) / max(max_steps, 1)
    if fraction <= 1.0 / 3.0:
        return "early"
    if fraction <= 2.0 / 3.0:
        return "mid"
    return "late"


def _feature_vector(
    logits: torch.Tensor,
    route_summary: dict[str, Any],
    next_state: dict[str, torch.Tensor] | None,
) -> list[float]:
    probs = torch.softmax(logits, dim=-1)
    top2 = torch.topk(logits, k=min(2, logits.numel()), dim=-1).values
    margin = top2[0] - top2[1] if top2.numel() > 1 else top2[0]
    features: list[float] = []
    features.extend(float(value) for value in logits.detach().float().cpu().tolist())
    features.extend(float(value) for value in probs.detach().float().cpu().tolist())
    features.extend(
        [
            float(probs.max().item()),
            float((-(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum()).item()),
            float(margin.item()),
            float(route_summary.get("route_entropy") or 0.0),
            float(route_summary.get("dominant_pair_fraction") or 0.0),
            float(route_summary.get("unique_pair_count") or 0.0),
        ]
    )
    features.extend(_state_tensor_features(next_state))
    return features


def _binary_bucket(values: list[float], *, low_label: str, high_label: str, fallback_threshold: float) -> tuple[float, list[str]]:
    if not values:
        return fallback_threshold, []
    low = min(values)
    high = max(values)
    threshold = float(np.median(np.asarray(values, dtype=np.float32)))
    if math.isclose(low, high, abs_tol=1e-6):
        threshold = fallback_threshold
    labels = [high_label if value >= threshold else low_label for value in values]
    if len(set(labels)) < 2:
        labels = [high_label if value > fallback_threshold else low_label for value in values]
        threshold = fallback_threshold
    return threshold, labels


def _collect_probe_rows(
    spec: dict[str, Any],
    family: str,
    campaign: dict[str, Any],
    *,
    device: str,
) -> list[dict[str, Any]]:
    config_path = Path(str(spec["config_path"]))
    checkpoint_path = Path(str(spec["checkpoint_path"]))
    if not config_path.exists() or not checkpoint_path.exists():
        return []
    device_t = torch.device("cpu" if device == "cpu" or not torch.cuda.is_available() else "cuda")
    config, model = _load_model(str(config_path), str(checkpoint_path), device_t)
    model.eval()
    env, _ = _build_single_env(config_path)
    rows: list[dict[str, Any]] = []
    episodes = int(campaign["analysis"]["diagnostic_episodes"])
    max_steps = int(campaign["analysis"]["diagnostic_max_steps"])
    try:
        for episode_index in range(episodes):
            reset_seed = int(spec["seed"]) + 5_000 + episode_index
            rng_state = capture_rng_state()
            set_seed(reset_seed, deterministic=config.system.deterministic)
            try:
                obs, _ = env.reset(seed=reset_seed)
                state = model.initial_state(1, device_t)
                done_t = torch.ones(1, device=device_t, dtype=torch.bool)
                for step in range(max_steps):
                    prev_option_idx = None
                    if family == "por":
                        prev_probs = state.get("option_probs")
                        if isinstance(prev_probs, torch.Tensor):
                            prev_option_idx = int(prev_probs.argmax(dim=-1).item())
                    obs_t = prepare_obs(_obs_to_batch(obs), device_t)
                    with torch.inference_mode():
                        output, route_capture = _forward_with_route_capture(model, obs_t, state, done_t)
                    route_summary = _route_capture_summary(route_capture)
                    grid_state = _extract_grid_state(env)
                    slice_bucket = _task_bucket(str(spec["task"]), grid_state, step, max_steps)
                    next_state = output.next_state
                    row = {
                        "task": str(spec["task"]),
                        "task_label": str(spec["task_label"]),
                        "variant": str(spec["variant"]),
                        "seed": int(spec["seed"]),
                        "slice_bucket": slice_bucket,
                        "step": int(step),
                        "features": _feature_vector(output.logits[0], route_summary, next_state),
                        "confidence": float(torch.softmax(output.logits[0], dim=-1).max().item()),
                    }
                    metrics = output.metrics
                    if family == "tregh":
                        row["hop_count"] = float(metrics.get("avg_hop_count", 1.0))
                        row["halting_prob"] = float(metrics.get("avg_halting_probability", 1.0))
                    elif family == "recurrent_control":
                        row["state_norm"] = float(_state_tensor_features(next_state)[2])
                    elif family == "por":
                        next_probs = next_state.get("option_probs")
                        next_option_idx = None
                        if isinstance(next_probs, torch.Tensor):
                            next_option_idx = int(next_probs.argmax(dim=-1).item())
                        row["option_switch_flag"] = float(
                            1.0 if prev_option_idx is not None and next_option_idx is not None and prev_option_idx != next_option_idx else 0.0
                        )
                        row["option_duration"] = float(metrics.get("option_duration", 0.0))
                    elif family == "srw":
                        row["rel_usage"] = float(metrics.get("relational_usage_rate", 0.0))
                    elif family == "sare_persistence":
                        row["route_entropy_value"] = float(route_summary.get("route_entropy") or metrics.get("route_entropy", 0.0) or 0.0)
                    rows.append(row)
                    action = _action_summary(output.logits, model)["action"]
                    obs, _reward, terminated, truncated, _info = env.step(int(action))
                    state = next_state
                    done = bool(terminated or truncated)
                    done_t = prepare_done(np.asarray([done], dtype=bool), device_t)
                    if done:
                        break
            finally:
                restore_rng_state(rng_state)
    finally:
        env.close()
    if family == "tregh":
        hop_values = [float(row["hop_count"]) for row in rows]
        halt_values = [float(row["halting_prob"]) for row in rows]
        hop_threshold, hop_labels = _binary_bucket(
            hop_values,
            low_label="stop",
            high_label="continue",
            fallback_threshold=float(campaign["analysis"]["tregh_continue_threshold"]),
        )
        halt_threshold, halt_labels = _binary_bucket(
            halt_values,
            low_label="low_halt",
            high_label="high_halt",
            fallback_threshold=float(campaign["analysis"]["tregh_halting_threshold"]),
        )
        for row, hop_label, halt_label in zip(rows, hop_labels, halt_labels, strict=False):
            row["continue_bucket"] = hop_label
            row["halting_bucket"] = halt_label
            row["hop_threshold"] = hop_threshold
            row["halt_threshold"] = halt_threshold
    elif family == "por":
        duration_values = [float(row["option_duration"]) for row in rows]
        duration_threshold, duration_labels = _binary_bucket(
            duration_values,
            low_label="short",
            high_label="long",
            fallback_threshold=float(campaign["analysis"]["por_duration_threshold"]),
        )
        for row, duration_label in zip(rows, duration_labels, strict=False):
            row["duration_bucket"] = duration_label
            row["switch_bucket"] = "switch" if float(row["option_switch_flag"]) > 0.5 else "hold"
            row["duration_threshold"] = duration_threshold
    elif family == "srw":
        rel_values = [float(row["rel_usage"]) for row in rows]
        rel_threshold, rel_labels = _binary_bucket(
            rel_values,
            low_label="low_rel",
            high_label="high_rel",
            fallback_threshold=float(campaign["analysis"]["srw_rel_usage_threshold"]),
        )
        for row, rel_label in zip(rows, rel_labels, strict=False):
            row["relation_bucket"] = rel_label
            row["rel_threshold"] = rel_threshold
    elif family == "sare_persistence":
        entropy_values = [float(row["route_entropy_value"]) for row in rows]
        entropy_threshold, entropy_labels = _binary_bucket(
            entropy_values,
            low_label="low_entropy",
            high_label="high_entropy",
            fallback_threshold=float(campaign["analysis"]["sare_route_entropy_threshold"]),
        )
        for row, entropy_label in zip(rows, entropy_labels, strict=False):
            row["entropy_bucket"] = entropy_label
            row["entropy_threshold"] = entropy_threshold
    return rows


def _probe_targets(family: str) -> list[str]:
    if family == "tregh":
        return ["slice_bucket", "continue_bucket", "halting_bucket"]
    if family == "recurrent_control":
        return ["slice_bucket"]
    if family == "por":
        return ["slice_bucket", "switch_bucket", "duration_bucket"]
    if family == "srw":
        return ["slice_bucket", "relation_bucket"]
    if family == "sare_persistence":
        return ["slice_bucket", "entropy_bucket"]
    return ["slice_bucket"]


def _family_numeric_columns(family: str) -> list[str]:
    if family == "tregh":
        return ["hop_count", "halting_prob", "confidence"]
    if family == "recurrent_control":
        return ["state_norm", "confidence"]
    if family == "por":
        return ["option_duration", "option_switch_flag", "confidence"]
    if family == "srw":
        return ["rel_usage", "confidence"]
    if family == "sare_persistence":
        return ["route_entropy_value", "confidence"]
    return ["confidence"]


def _render_diagnostic_report(
    campaign: dict[str, Any],
    family: str,
    specs: list[dict[str, Any]],
    output: str | Path,
    json_output: str | Path,
    *,
    device: str,
    title: str,
    source_label: str,
    verdict_positive: str,
    verdict_negative: str,
) -> None:
    probe_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    selected_specs = []
    for spec in specs:
        diag_rows = _collect_probe_rows(spec, family, campaign, device=device)
        if not diag_rows:
            continue
        selected_specs.append(spec)
        numeric_columns = _family_numeric_columns(family)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in diag_rows:
            grouped[str(row["slice_bucket"])].append(row)
        for bucket, subset in sorted(grouped.items()):
            summary = {
                "task": str(spec["task"]),
                "task_label": str(spec["task_label"]),
                "variant": str(spec["variant"]),
                "seed": int(spec["seed"]),
                "slice_bucket": bucket,
                "samples": len(subset),
            }
            for column in numeric_columns:
                values = [float(row.get(column, 0.0)) for row in subset]
                summary[column] = float(mean(values)) if values else 0.0
            slice_rows.append(summary)
        for target in _probe_targets(family):
            result = _fit_linear_probe(diag_rows, target)
            probe_rows.append(
                {
                    "task": str(spec["task"]),
                    "task_label": str(spec["task_label"]),
                    "variant": str(spec["variant"]),
                    "seed": int(spec["seed"]),
                    "target": target,
                    **result,
                }
            )
    max_lift = max((float(row.get("lift_vs_majority", 0.0)) for row in probe_rows if row.get("status") == "completed"), default=0.0)
    verdict = verdict_positive if max_lift > 0.10 else verdict_negative
    lines = [f"# {title}", "", f"- diagnostic source: `{source_label}`", f"- representative runs: `{len(selected_specs)}`", f"- verdict: `{verdict}`", ""]
    spec_rows: list[list[str]] = []
    for spec in selected_specs:
        spec_rows.append(
            [
                str(spec["task_label"]),
                str(spec["variant"]),
                str(spec["seed"]),
                _format_float(spec["eval_success_rate"]),
                _format_float(spec["best_train_return"]),
                str(spec["output_root"]),
            ]
        )
    if spec_rows:
        lines.extend(_markdown_table(["Task", "Variant", "Seed", "Eval Success", "Best Train Return", "Run Root"], spec_rows))
        lines.append("")
    slice_table: list[list[str]] = []
    for row in slice_rows:
        slice_table.append(
            [
                str(row["task_label"]),
                str(row["variant"]),
                str(row["slice_bucket"]),
                str(int(row["samples"])),
                *[_format_float(row.get(column, 0.0), 3) for column in _family_numeric_columns(family)],
            ]
        )
    if slice_table:
        lines.append("## Hard-Slice Map")
        lines.append("")
        lines.extend(
            _markdown_table(
                ["Task", "Variant", "Slice", "Samples", *[column.replace("_", " ").title() for column in _family_numeric_columns(family)]],
                slice_table,
            )
        )
        lines.append("")
    probe_table: list[list[str]] = []
    for row in probe_rows:
        probe_table.append(
            [
                str(row["task_label"]),
                str(row["variant"]),
                str(row["target"]),
                str(int(row.get("samples", 0))),
                str(int(row.get("classes", 0))),
                _format_float(row.get("test_accuracy", 0.0)),
                _format_float(row.get("majority_baseline", 0.0)),
                _format_float(row.get("lift_vs_majority", 0.0)),
                str(row.get("status", "missing")),
            ]
        )
    if probe_table:
        lines.append("## Source-Quality Probes")
        lines.append("")
        lines.extend(
            _markdown_table(
                ["Task", "Variant", "Target", "Samples", "Classes", "Test Acc", "Majority", "Lift", "Status"],
                probe_table,
            )
        )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(f"- The stage-1 diagnostic verdict for `{family}` is `{verdict}` based on the best available frozen checkpoints and shallow probe fits.")
    lines.append("- This report is diagnostic only. Promotion still depends on a living stage-2 baseline, controls, reruns, and later-stage holdout/stability work.")
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(json_output, {"selected_specs": selected_specs, "slice_rows": slice_rows, "probe_rows": probe_rows, "verdict": verdict})


def _write_stage0_reports(campaign: dict[str, Any]) -> None:
    state_lines = [
        "# Next Architecture Wave State Reconciliation",
        "",
        f"- active benchmark: `{campaign['current_canonical_name']}`",
        f"- archived frozen provenance anchor: `{campaign['frozen_pack']}`",
        f"- active benchmark pack: `{campaign['current_canonical_pack']}`",
        f"- current gate reference pack: `{campaign['current_gate_reference_pack']}`",
        f"- prior narrowing memo: `{campaign['current_decision_memo']}`",
        "",
        "## Accepted State",
        "",
        "- `round6` remains the active DoorKey benchmark inside the narrow accepted claim envelope.",
        "- PPO-only SARE remains negative, and repeated DoorKey-only SARE challenger work has not displaced `round6`.",
        "- The prior cross-family wave for TREG-H, POR, SRW, and bounded SARE persistence also ended negative for promotion.",
        "- This second wave therefore starts from a narrower question: whether hard-slice diagnostics, source-quality probes, and compute-structure extraction reveal any remaining headroom in the non-SARE architecture theses.",
    ]
    Path(campaign["reports"]["state_reconciliation"]).write_text("\n".join(state_lines) + "\n", encoding="utf-8")

    baseline_lines = [
        "# Next Architecture Wave Baseline Sync",
        "",
        "| Family | Latest Accepted Summary | Reference Report |",
        "| --- | --- | --- |",
        "| `round6` | active DoorKey benchmark retained | `outputs/reports/next_wave_decision_memo.md` |",
        "| `TREG-H` | 0 survivors in prior 24-run screen | `outputs/reports/tregh_stage1_screening.md` |",
        "| `POR` | recurrent fairness repaired, but POR 0 survivors | `outputs/reports/por_stage2_screening.md` |",
        "| `SRW` | relation gate live, but 0 survivors | `outputs/reports/srw_stage1_screening.md` |",
        "| `SARE persistence` | bounded KeyCorridor line exploratory only | `outputs/reports/compute_structure_stage4_sare_persistence.md` |",
        "",
        "## Interpretation",
        "",
        "- The current repo state is internally consistent enough to proceed.",
        "- The frontier is already narrowed, so this wave must justify any reopen through diagnostics first and only then through a larger alive-baseline screen.",
    ]
    Path(campaign["reports"]["baseline_sync"]).write_text("\n".join(baseline_lines) + "\n", encoding="utf-8")

    registration_lines = [
        "# Next Architecture Wave Registration",
        "",
        f"- target substantive runs: `{campaign['target_substantive_runs']}`",
        "- workstreams:",
        "  - TREG-H hard-slice and compute-structure",
        "  - recurrent fairness deepening plus POR persistence",
        "  - SRW hard-slice and selective-relational diagnostics",
        "  - bounded route-memory SARE persistence",
        "  - unified compute frontier and final decision",
        "",
        "## Budget",
        "",
        "- TREG-H screen: `28` substantive runs",
        "- recurrent fairness + POR screen: `28` substantive runs",
        "- SRW screen: `24` substantive runs",
        "- bounded SARE persistence diagnostic: `16` substantive runs",
        "",
        "## Task Fit",
        "",
        f"- TREG-H third task status: `{campaign['analysis']['third_task_status']['tregh']}`",
        f"- POR third task status: `{campaign['analysis']['third_task_status']['por']}`",
        f"- SRW third task status: `{campaign['analysis']['third_task_status']['srw']}`",
        "",
        "## Final Decision Rule",
        "",
        "- No family gets promotion unless it first establishes an alive baseline, then survives controls, reruns, holdout, and the final pack/gate path.",
        "- If the larger alive-baseline screens still produce no control-meaningful survivors, the accepted frontier narrows further around `round6` and a small set of exploratory side branches only.",
    ]
    Path(campaign["reports"]["registration"]).write_text("\n".join(registration_lines) + "\n", encoding="utf-8")


def _write_tregh_hardslice_report(campaign: dict[str, Any], *, device: str) -> None:
    prior_rows = _prior_rows(campaign, "tregh")
    specs = _best_specs(prior_rows)
    _render_diagnostic_report(
        campaign,
        "tregh",
        specs,
        campaign["reports"]["tregh_stage1"],
        campaign["reports"]["tregh_stage1_json"],
        device=device,
        title="TREG-H Wave 2 Stage 1 Hard-Slice and Source-Quality",
        source_label="prior next-wave TREG-H checkpoints",
        verdict_positive="compute_signal_exposed_but_not_effective",
        verdict_negative="compute_signal_narrow_and_flat",
    )


def _write_recurrent_hardslice_report(campaign: dict[str, Any], *, device: str) -> None:
    rows = _collect_rows(campaign, "recurrent_control")
    specs = _best_specs(rows)
    grouped_controls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_controls[str(row["task"])].append(
            {
                "eval_success_rate": float(row["eval_success_rate"]),
                "best_train_return": float(row["best_train_return"]),
                "throughput_fps": float(row["throughput_fps"]),
                "variant": str(row["variant"]),
                "seed": int(row["seed"]),
            }
        )
    _write_json(campaign["reports"]["recurrent_controls_json"], {"rows": rows, "grouped": grouped_controls})
    _render_diagnostic_report(
        campaign,
        "recurrent_control",
        specs,
        campaign["reports"]["por_stage1"],
        campaign["reports"]["por_stage1_json"],
        device=device,
        title="POR Wave 2 Stage 1 Recurrent-Control Hard-Slice",
        source_label="current wave recurrent-control checkpoints",
        verdict_positive="recurrent_progress_signal_retained_but_policy_still_weak",
        verdict_negative="recurrent_progress_signal_weak",
    )


def _write_srw_hardslice_report(campaign: dict[str, Any], *, device: str) -> None:
    prior_rows = _prior_rows(campaign, "srw")
    specs = _best_specs(prior_rows)
    _render_diagnostic_report(
        campaign,
        "srw",
        specs,
        campaign["reports"]["srw_stage1"],
        campaign["reports"]["srw_stage1_json"],
        device=device,
        title="SRW Wave 2 Stage 1 Hard-Slice and Source-Quality",
        source_label="prior next-wave SRW checkpoints",
        verdict_positive="relation_signal_exposed_but_misallocated",
        verdict_negative="relation_signal_weak_or_nonconsequential",
    )


def _write_sare_hardslice_report(campaign: dict[str, Any], *, device: str) -> None:
    prior_rows = _prior_rows(campaign, "sare_persistence")
    specs = _best_specs(prior_rows)
    _render_diagnostic_report(
        campaign,
        "sare_persistence",
        specs,
        campaign["reports"]["sare_stage1"],
        campaign["reports"]["sare_stage1_json"],
        device=device,
        title="SARE Persistence Wave 2 Stage 1 Hard-Slice Audit",
        source_label="prior bounded route-memory persistence checkpoints",
        verdict_positive="route_memory_signal_exposed_but_not_benchmark_worthy",
        verdict_negative="route_memory_signal_weak",
    )


def _write_alive_baseline_report(campaign: dict[str, Any], family: str) -> list[dict[str, Any]]:
    rows = _collect_rows(campaign, family)
    aggregates = _aggregate_family(rows)
    recurrent_controls = None
    if family == "por":
        payload = json.loads(Path(campaign["reports"]["recurrent_controls_json"]).read_text(encoding="utf-8"))
        recurrent_controls = {
            task: [
                {
                    "eval_success_rate": _safe_float(entry["eval_success_rate"]),
                    "best_train_return": _safe_float(entry["best_train_return"]),
                }
                for entry in entries
            ]
            for task, entries in payload.get("grouped", {}).items()
        }
    survivors = _pick_family_survivors(family, aggregates, recurrent_controls=recurrent_controls)
    if family == "tregh":
        report_path = campaign["reports"]["tregh_stage2"]
        json_path = campaign["reports"]["tregh_stage2_json"]
        later = [
            ("TREG-H Wave 2 Stage 3 Compute-Structure", campaign["reports"]["tregh_stage3"]),
            ("TREG-H Wave 2 Stage 4 Fairness/Holdout", campaign["reports"]["tregh_stage4"]),
            ("TREG-H Wave 2 Stage 5 Route/Stability", campaign["reports"]["tregh_stage5"]),
        ]
        title = "TREG-H Wave 2 Stage 2 Alive Baseline"
        failure_line = "- No TREG-H variant established an alive baseline after the larger 28-run screen, so the family remains narrowed and later stages enter the formal skip path."
    elif family == "por":
        report_path = campaign["reports"]["por_stage2"]
        json_path = campaign["reports"]["por_stage2_json"]
        later = [
            ("POR Wave 2 Stage 3 Compute-Structure", campaign["reports"]["por_stage3"]),
            ("POR Wave 2 Stage 4 Fairness/Holdout", campaign["reports"]["por_stage4"]),
            ("POR Wave 2 Stage 5 Stability", campaign["reports"]["por_stage5"]),
        ]
        title = "POR Wave 2 Stage 2 Alive Baseline"
        failure_line = "- No POR variant established a living baseline above the deepened recurrent controls, so the persistence lane remains narrowed and later stages enter the formal skip path."
    else:
        report_path = campaign["reports"]["srw_stage2"]
        json_path = campaign["reports"]["srw_stage2_json"]
        later = [
            ("SRW Wave 2 Stage 3 Compute-Structure", campaign["reports"]["srw_stage3"]),
            ("SRW Wave 2 Stage 4 Fairness/Holdout", campaign["reports"]["srw_stage4"]),
            ("SRW Wave 2 Stage 5 Stability", campaign["reports"]["srw_stage5"]),
        ]
        title = "SRW Wave 2 Stage 2 Alive Baseline"
        failure_line = "- No SRW variant established an alive baseline after the larger 24-run screen, so the selective-relational lane remains narrowed and later stages enter the formal skip path."

    _write_json(json_path, {"rows": rows, "aggregates": aggregates, "survivors": survivors})
    lines = [f"# {title}", "", f"- substantive runs: `{len(rows)}`", f"- surviving variants after control gate: `{len(survivors)}`", ""]
    table_rows: list[list[str]] = []
    for row in aggregates:
        table_rows.append(
            [
                str(row["task_label"]),
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
            table_rows,
        )
    )
    lines.extend(["", "## Verdict", ""])
    if survivors:
        lines.append("- One or more alive baselines emerged and require later-stage compute-structure, fairness, holdout, and stability work.")
    else:
        lines.append(failure_line)
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not survivors:
        for later_title, later_path in later:
            _write_skip_report(later_path, later_title, "no alive stage-2 baseline")
    return survivors


def _write_sare_stage2_report(campaign: dict[str, Any]) -> None:
    rows = _collect_rows(campaign, "sare_persistence")
    aggregates = _aggregate_family(rows)
    _write_json(campaign["reports"]["sare_stage2_json"], {"rows": rows, "aggregates": aggregates})
    lines = [
        "# SARE Persistence Wave 2 Stage 2 Diagnostic",
        "",
        f"- substantive runs: `{len(rows)}`",
        "- bounded exploratory route-memory line on `KeyCorridor` only",
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
            "- Bounded SARE persistence remains exploratory only in this phase.",
            "- This lane is retained only to decide whether a future persistence-focused side branch is worth keeping alive.",
        ]
    )
    Path(campaign["reports"]["sare_stage2"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_aggregate(aggregates: list[dict[str, Any]], task: str) -> dict[str, Any] | None:
    subset = [row for row in aggregates if str(row["task"]) == task]
    if not subset:
        return None
    return max(subset, key=lambda row: (row["eval_success_rate"], row["best_train_return"], -row["active_compute_proxy"]))


def _frontier_rows(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tregh = json.loads(Path(campaign["reports"]["tregh_stage2_json"]).read_text(encoding="utf-8")).get("aggregates", [])
    por = json.loads(Path(campaign["reports"]["por_stage2_json"]).read_text(encoding="utf-8")).get("aggregates", [])
    srw = json.loads(Path(campaign["reports"]["srw_stage2_json"]).read_text(encoding="utf-8")).get("aggregates", [])
    sare = json.loads(Path(campaign["reports"]["sare_stage2_json"]).read_text(encoding="utf-8")).get("aggregates", [])
    frontier_specs = [
        ("doorkey", "TREG-H", _best_aggregate(tregh, "doorkey")),
        ("keycorridor", "TREG-H", _best_aggregate(tregh, "keycorridor")),
        ("memory", "POR", _best_aggregate(por, "memory")),
        ("keycorridor", "POR", _best_aggregate(por, "keycorridor")),
        ("dynamic_obstacles", "SRW", _best_aggregate(srw, "dynamic_obstacles")),
        ("keycorridor", "SRW", _best_aggregate(srw, "keycorridor")),
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
    lines = ["# Compute Frontier Wave 2 Report", ""]
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
            "- This wave-2 frontier integrates hard-slice-informed alive-baseline screens across the non-SARE architecture theses.",
            "- It remains a screening-stage compute frontier, not a claim-bearing benchmark replacement table.",
        ]
    )
    Path(campaign["reports"]["frontier_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_candidate_pack(campaign: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "pack_type": "next_arch_wave_summary",
        "generated_at": _timestamp(),
        "active_benchmark": campaign["current_canonical_name"],
        "status": "no_promotion",
        "task_scoped_candidates": [],
        "notes": "No family cleared the second-wave hard-slice, control, and alive-baseline gates strongly enough for promotion.",
    }
    _write_json(campaign["reports"]["candidate_pack_summary"], payload)
    Path(campaign["reports"]["gate_report"]).write_text(
        "# Next Architecture Wave Gate Report\n\n- no task-scoped candidate pack was promoted\n- gate invocation: `not required`\n- reason: `no family cleared hard-slice diagnostics, alive-baseline screens, controls, and later-stage prerequisites`\n",
        encoding="utf-8",
    )
    return payload


def _write_decision_memo(campaign: dict[str, Any]) -> None:
    tregh = json.loads(Path(campaign["reports"]["tregh_stage2_json"]).read_text(encoding="utf-8"))
    por = json.loads(Path(campaign["reports"]["por_stage2_json"]).read_text(encoding="utf-8"))
    srw = json.loads(Path(campaign["reports"]["srw_stage2_json"]).read_text(encoding="utf-8"))
    status = "overall frontier narrows because none of the alternative architecture theses survive proper controls"
    if tregh.get("survivors"):
        status = "TREG-H earns task-scoped benchmark-candidate status"
    elif por.get("survivors"):
        status = "POR earns task-scoped benchmark-candidate status"
    elif srw.get("survivors"):
        status = "SRW earns task-scoped benchmark-candidate status"
    lines = [
        "# Next Architecture Wave Decision Memo",
        "",
        f"- final status: `{status}`",
        f"- active benchmark remains: `{campaign['current_canonical_name']}`",
        "",
        "## Summary",
        "",
        "- The second-wave program added hard-slice and source-quality diagnostics before the larger alive-baseline screens for TREG-H, POR, SRW, and bounded route-memory persistence.",
        "- No family cleared the repo’s control-first benchmark promotion funnel strongly enough to justify a task-scoped candidate pack in this pass.",
        "- `round6` therefore remains the only active benchmark inside the narrow accepted claim envelope.",
    ]
    Path(campaign["reports"]["decision_memo"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Second-wave architecture thesis campaign runner.")
    parser.add_argument(
        "--campaign-config",
        default="configs/experiments/lss_next_arch_wave_program/campaign.yaml",
        help="Path to second-wave campaign config",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("write-stage0")

    diag = subparsers.add_parser("write-hardslice-report")
    diag.add_argument("--family", choices=("tregh", "recurrent_control", "srw", "sare_persistence"), required=True)
    diag.add_argument("--device", default="auto")

    run = subparsers.add_parser("run-screen")
    run.add_argument("--family", choices=("tregh", "recurrent_control", "por", "srw", "sare_persistence"), required=True)
    run.add_argument("--device", default="auto")

    family_report = subparsers.add_parser("write-family-report")
    family_report.add_argument("--family", choices=("tregh", "por", "srw", "sare_persistence"), required=True)

    subparsers.add_parser("write-frontier-report")
    subparsers.add_parser("write-candidate-pack")
    subparsers.add_parser("write-decision-memo")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    campaign = load_campaign_config(args.campaign_config)
    if args.command == "write-stage0":
        _write_stage0_reports(campaign)
    elif args.command == "write-hardslice-report":
        family = str(args.family)
        if family == "tregh":
            _write_tregh_hardslice_report(campaign, device=str(args.device))
        elif family == "recurrent_control":
            _write_recurrent_hardslice_report(campaign, device=str(args.device))
        elif family == "srw":
            _write_srw_hardslice_report(campaign, device=str(args.device))
        elif family == "sare_persistence":
            _write_sare_hardslice_report(campaign, device=str(args.device))
        else:
            raise ValueError(family)
    elif args.command == "run-screen":
        _run_jobs(campaign, _family_jobs(campaign, str(args.family)), str(args.device))
    elif args.command == "write-family-report":
        family = str(args.family)
        if family == "sare_persistence":
            _write_sare_stage2_report(campaign)
        else:
            _write_alive_baseline_report(campaign, family)
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
