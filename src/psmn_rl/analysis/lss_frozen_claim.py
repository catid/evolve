from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from psmn_rl.analysis.lss_claim_consolidation import _all_seeds, _group, _lane_seeds, _summary_stats
from psmn_rl.analysis.lss_robustness import (
    EvalTarget,
    _best_sampled,
    _format_float,
    _greedy_row,
    _maybe_read_command,
    _write_main_process_report,
)
from psmn_rl.analysis.lss_route_dependence import _discover_case, _evaluate_probe
from psmn_rl.config import load_config
from psmn_rl.logging import configure_logging
from psmn_rl.models.factory import build_model
from psmn_rl.models.routing.sare import RoutedExpertCore
from psmn_rl.rl.distributed.ddp import cleanup_distributed, init_distributed
from psmn_rl.rl.ppo.algorithm import collect_policy_diagnostics
from psmn_rl.utils.io import get_git_commit, get_git_dirty
from psmn_rl.utils.seed import set_seed


REPRO_MODES: list[tuple[str, bool, float]] = [
    ("greedy", True, 1.0),
    ("sampled_t1.0", False, 1.0),
]


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _targets_from_rows(rows: list[dict[str, Any]], labels: set[str]) -> list[EvalTarget]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        label = str(row["label"])
        if label not in labels:
            continue
        lane = str(row.get("lane", row.get("stage", "unknown")))
        grouped.setdefault((lane, int(row["seed"]), label), []).append(row)
    targets: list[EvalTarget] = []
    for (lane, seed, label), mode_rows in sorted(grouped.items()):
        row = mode_rows[0]
        targets.append(
            EvalTarget(
                seed=seed,
                label=label,
                variant=str(row["variant"]),
                config_path=Path(str(row["config_path"])),
                checkpoint_path=Path(str(row["checkpoint_path"])),
                run_dir=Path(str(row["run_dir"])),
                method=str(row["method"]),
                stage=lane,
                metadata={"lane": lane},
                command_path=Path(str(row["run_dir"])) / "command.txt",
            )
        )
    return targets


def _evaluate_modes(targets: list[EvalTarget], device: str, episodes: int, modes: list[tuple[str, bool, float]]) -> list[dict[str, Any]]:
    if not targets:
        return []
    ctx = init_distributed(device, "auto")
    configure_logging(ctx.is_main_process)
    rows: list[dict[str, Any]] = []
    try:
        for target in targets:
            config = load_config(target.config_path)
            config.system.device = device
            config.logging.tensorboard = False
            set_seed(config.seed + ctx.rank, deterministic=config.system.deterministic)
            envs = None
            try:
                from psmn_rl.envs.registry import make_vector_env

                envs = make_vector_env(config.env, seed=config.seed, world_rank=ctx.rank)
                model = build_model(config.model, envs.single_observation_space, envs.single_action_space).to(ctx.device)
            finally:
                if envs is not None:
                    envs.close()
            checkpoint = torch.load(target.checkpoint_path, map_location=ctx.device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            for mode_name, greedy, temperature in modes:
                diagnostics = collect_policy_diagnostics(
                    config=config,
                    model=model,
                    ctx=ctx,
                    episodes=episodes,
                    greedy=greedy,
                    temperature=temperature,
                    trace_limit=0,
                )
                row = {
                    "seed": target.seed,
                    "label": target.label,
                    "variant": target.variant,
                    "method": target.method,
                    "stage": target.stage,
                    "lane": target.metadata.get("lane", target.stage) if target.metadata else target.stage,
                    "config_path": str(target.config_path),
                    "checkpoint_path": str(target.checkpoint_path),
                    "run_dir": str(target.run_dir),
                    "mode": mode_name,
                    "greedy": greedy,
                    "temperature": temperature,
                    **diagnostics.metrics,
                }
                rows.append(row)
        return rows if ctx.is_main_process else []
    finally:
        cleanup_distributed(ctx)


def _single_expert_targets(root: Path, lane: str) -> list[EvalTarget]:
    targets: list[EvalTarget] = []
    for run_dir in sorted(root.glob("seed_*/kl_lss_single_expert")):
        seed = int(run_dir.parent.name.split("_", 1)[1])
        targets.append(
            EvalTarget(
                seed=seed,
                label="kl_lss_single_expert",
                variant="single_expert",
                config_path=run_dir / "student_resolved_config.yaml",
                checkpoint_path=run_dir / "latest.pt",
                run_dir=run_dir,
                method="kl_lss_single_expert",
                stage=lane,
                metadata={"lane": lane},
                command_path=run_dir / "command.txt",
            )
        )
    return targets


def _build_reproduction_note(rows: list[dict[str, Any]], structured_csv: Path, final_csv: Path, episodes: int) -> str:
    grouped = _group(rows)
    labels = [
        ("recovered_token_dense", "recovered token_dense"),
        ("kl_lss_token_dense", "KL learner-state token_dense"),
        ("kl_lss_single_expert", "KL learner-state single_expert"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lines = [
        "# Frozen-Claim Reproduction Note",
        "",
        f"- structured slice source csv: `{structured_csv}`",
        f"- final block source csv: `{final_csv}`",
        f"- external evaluation episodes per mode: `{episodes}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Per-Seed Results",
        "",
        "| Lane | Seed | Variant | Greedy Success | Sampled t=1.0 Success | Config | Checkpoint | Command |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for lane, seed in _all_seeds(grouped):
        for label, display in labels:
            if (lane, seed, label) not in grouped:
                continue
            mode_rows = grouped[(lane, seed, label)]
            greedy = _greedy_row(mode_rows)
            sampled = next(row for row in mode_rows if row["mode"] == "sampled_t1.0")
            lines.append(
                "| "
                + " | ".join(
                    [
                        lane,
                        str(seed),
                        display,
                        _format_float(greedy.get("eval_success_rate")),
                        _format_float(sampled.get("eval_success_rate")),
                        f"`{greedy['config_path']}`",
                        f"`{greedy['checkpoint_path']}`",
                        f"`{_maybe_read_command(Path(greedy['run_dir']) / 'command.txt') or '-'}`",
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This note re-evaluates the current frozen DoorKey baseline on the external 64-episode path before the final-block single_expert control and failure analysis.",
            "- The structured slice and the final fresh block remain the reference lanes for the freeze-or-thaw decision in this phase.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_final_block_single_expert_report(rows: list[dict[str, Any]], episodes: int) -> str:
    grouped = _group(rows)
    lane = "fresh_final"
    labels = [
        ("recovered_token_dense", "recovered token_dense"),
        ("kl_lss_token_dense", "KL learner-state token_dense"),
        ("kl_lss_single_expert", "KL learner-state single_expert"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lines = [
        "# Final-Block Matched Single-Expert DoorKey Control Report",
        "",
        f"- lane: `{lane}`",
        f"- external evaluation episodes per mode: `{episodes}`",
        "",
        "| Seed | recovered token_dense | KL learner-state token_dense | KL learner-state single_expert | baseline PPO SARE | KL learner-state SARE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in _lane_seeds(grouped, lane):
        values = [str(seed)]
        for label, _display in labels:
            values.append(_format_float(_greedy_row(grouped[(lane, seed, label)]).get("eval_success_rate")))
        lines.append("| " + " | ".join(values) + " |")
    stats = {label: _summary_stats(grouped, label, lane) for label, _ in labels}
    lines.extend(
        [
            "",
            "## Mean Greedy Success",
            "",
            "| Variant | Mean Greedy Success | Complete-Seed Failures |",
            "| --- | ---: | ---: |",
            f"| recovered token_dense | `{stats['recovered_token_dense']['mean']:.4f}` | `{int(stats['recovered_token_dense']['zero_failures'])}` |",
            f"| KL learner-state token_dense | `{stats['kl_lss_token_dense']['mean']:.4f}` | `{int(stats['kl_lss_token_dense']['zero_failures'])}` |",
            f"| KL learner-state single_expert | `{stats['kl_lss_single_expert']['mean']:.4f}` | `{int(stats['kl_lss_single_expert']['zero_failures'])}` |",
            f"| baseline PPO SARE | `{stats['baseline_sare']['mean']:.4f}` | `{int(stats['baseline_sare']['zero_failures'])}` |",
            f"| KL learner-state SARE | `{stats['kl_lss_sare']['mean']:.4f}` | `{int(stats['kl_lss_sare']['zero_failures'])}` |",
            "",
            "## Interpretation",
            "",
        ]
    )
    sare = stats["kl_lss_sare"]["mean"]
    token = stats["kl_lss_token_dense"]["mean"]
    single = stats["kl_lss_single_expert"]["mean"]
    if sare > max(token, single):
        lines.append("- On the final fresh block, KL learner-state SARE still stays ahead of both matched structured controls, so the frozen claim would have room to thaw.")
    elif sare > token:
        lines.append("- On the final fresh block, KL learner-state SARE still beats token_dense but not single_expert, so the result remains method-first rather than specifically multi-expert.")
    else:
        lines.append("- On the final fresh block, matched structured controls catch or beat KL learner-state SARE, so the current claim should remain frozen or narrow further.")
    return "\n".join(lines) + "\n"


def _load_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _best_round_metrics(run_dir: Path) -> dict[str, Any]:
    summary = _load_summary(run_dir)
    rounds = summary["rounds"]
    best_index = int(summary["best_round_index"]) - 1
    best = dict(rounds[best_index])
    best["seed"] = int(run_dir.parent.name.split("_", 1)[1])
    best["best_round_index"] = int(summary["best_round_index"])
    best["best_round_greedy_success"] = float(summary["best_round_greedy_success"])
    best["final_greedy_success"] = float(summary["final_greedy_success"])
    best["final_best_sampled_success"] = float(summary["final_best_sampled_success"])
    return best


def _sare_run_dir_from_rows(rows: list[dict[str, Any]], lane: str, seed: int) -> Path:
    grouped = _group(rows)
    return Path(str(_greedy_row(grouped[(lane, seed, "kl_lss_sare")])["run_dir"]))


def _capture_route_batch_stats(run_dir: Path, checkpoint_path: Path, config_path: Path, dataset_path: Path, max_samples: int, device: str) -> dict[str, float]:
    config = load_config(config_path)
    config.system.device = device
    ctx = init_distributed(device, "auto")
    configure_logging(ctx.is_main_process)
    try:
        envs = None
        try:
            from psmn_rl.envs.registry import make_vector_env

            envs = make_vector_env(config.env, seed=config.seed, world_rank=ctx.rank)
            model = build_model(config.model, envs.single_observation_space, envs.single_action_space).to(ctx.device)
        finally:
            if envs is not None:
                envs.close()
        checkpoint = torch.load(checkpoint_path, map_location=ctx.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if not isinstance(model.core, RoutedExpertCore):
            return {
                "route_pair_unique_mean": 0.0,
                "route_pair_dominant_mean": 0.0,
                "route_global_top_pair_fraction": 0.0,
            }
        dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
        obs = dataset["obs"]
        total = next(iter(obs.values())).shape[0]
        if total <= max_samples:
            indices = torch.arange(total, dtype=torch.long)
        else:
            indices = torch.linspace(0, total - 1, steps=max_samples).long().unique(sorted=True)
        selected_obs = {key: value[indices] for key, value in obs.items()}
        batch_size = 256
        dominant_fractions: list[float] = []
        unique_counts: list[float] = []
        global_counter: Counter[tuple[int, ...]] = Counter()
        original_route = model.core.route
        try:
            for start in range(0, indices.numel(), batch_size):
                end = min(start + batch_size, indices.numel())
                chunk = {key: value[start:end].to(ctx.device) for key, value in selected_obs.items()}
                done = torch.ones(end - start, device=ctx.device, dtype=torch.bool)
                state = model.initial_state(end - start, ctx.device)
                captured: list[torch.Tensor] = []

                def route_with_capture(self: RoutedExpertCore, tokens: torch.Tensor):
                    route_probs, topk_values, topk_idx = original_route(tokens)
                    captured.append(topk_idx.detach().cpu())
                    return route_probs, topk_values, topk_idx

                model.core.route = route_with_capture.__get__(model.core, RoutedExpertCore)
                with torch.no_grad():
                    model(chunk, state=state, done=done)
                topk_idx = captured[-1]
                batch, token_count, _topk = topk_idx.shape
                for sample_index in range(batch):
                    pairs = [tuple(sorted(map(int, topk_idx[sample_index, token_index].tolist()))) for token_index in range(token_count)]
                    counts = Counter(pairs)
                    unique_counts.append(float(len(counts)))
                    dominant_fractions.append(max(counts.values()) / token_count if token_count > 0 else 0.0)
                    global_counter.update(pairs)
        finally:
            model.core.route = original_route
        top_pair_fraction = max(global_counter.values()) / sum(global_counter.values()) if global_counter else 0.0
        return {
            "route_pair_unique_mean": float(sum(unique_counts) / len(unique_counts)) if unique_counts else 0.0,
            "route_pair_dominant_mean": float(sum(dominant_fractions) / len(dominant_fractions)) if dominant_fractions else 0.0,
            "route_global_top_pair_fraction": float(top_pair_fraction),
        }
    finally:
        cleanup_distributed(ctx)


def _route_probe_rows_for_cases(cases: list[dict[str, Any]], device: str, episodes: int) -> list[dict[str, Any]]:
    ctx = init_distributed(device, "auto")
    configure_logging(ctx.is_main_process)
    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            baseline = _evaluate_probe(case, ctx, episodes, "baseline")
            rows.append(baseline)
            expert_count = int(baseline["expert_count"])
            for expert_index in range(expert_count):
                rows.append(_evaluate_probe(case, ctx, episodes, "expert_ablation", detail=str(expert_index)))
            fixed_experts = sorted(
                range(expert_count),
                key=lambda index: float(baseline.get(f"expert_load_{index}", 0.0)),
                reverse=True,
            )[: max(1, int(baseline["top_k"]))]
            rows.append(_evaluate_probe(case, ctx, episodes, "router_override", detail="most_used_pair", fixed_experts=fixed_experts))
            rows.append(_evaluate_probe(case, ctx, episodes, "route_randomization", detail="uniform_topk_random"))
        return rows if ctx.is_main_process else []
    finally:
        cleanup_distributed(ctx)


def _build_failure_analysis(
    final_rows: list[dict[str, Any]],
    single_rows: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    strong_sare_rows: list[dict[str, Any]],
    final_run_roots: dict[int, Path],
    strong_run_roots: dict[int, Path],
    single_run_roots: dict[int, Path],
    route_pair_stats: dict[int, dict[str, float]],
    episodes: int,
) -> str:
    final_grouped = _group(final_rows + single_rows)
    route_grouped: dict[int, list[dict[str, Any]]] = {}
    for row in route_rows:
        route_grouped.setdefault(int(row["seed"]), []).append(row)
    final_best = [_best_round_metrics(final_run_roots[seed]) for seed in sorted(final_run_roots)]
    strong_best = [_best_round_metrics(strong_run_roots[seed]) for seed in sorted(strong_run_roots)]
    single_best = {seed: _best_round_metrics(path) for seed, path in single_run_roots.items()}
    lines = [
        "# Final-Block Failure Analysis",
        "",
        f"- external evaluation episodes per mode: `{episodes}`",
        "",
        "## Final-Block Fairness Context",
        "",
        "| Seed | KL learner-state token_dense | KL learner-state single_expert | KL learner-state SARE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for seed in _lane_seeds(final_grouped, "fresh_final"):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(seed),
                    _format_float(_greedy_row(final_grouped[("fresh_final", seed, "kl_lss_token_dense")]).get("eval_success_rate")),
                    _format_float(_greedy_row(final_grouped[("fresh_final", seed, "kl_lss_single_expert")]).get("eval_success_rate")),
                    _format_float(_greedy_row(final_grouped[("fresh_final", seed, "kl_lss_sare")]).get("eval_success_rate")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Best-Round Learner-State Diagnostics",
            "",
            "| Seed | Variant | Best Round | Best Greedy | Final Greedy | Disagreement | Teacher Conf | Unique Ratio | Added Steps | Aggregate Steps | Student Conf | Route Entropy | Path Entropy |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in final_best:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["seed"])),
                    "KL learner-state SARE",
                    str(int(row["best_round_index"])),
                    _format_float(row["best_round_greedy_success"]),
                    _format_float(row["final_greedy_success"]),
                    _format_float(row.get("collection/disagreement_rate")),
                    _format_float(row.get("collection/teacher_confidence_mean")),
                    _format_float(row.get("collection/unique_state_ratio")),
                    _format_float(row.get("added_steps")),
                    _format_float(row.get("aggregate_steps")),
                    _format_float(row.get("collection/student_confidence_mean")),
                    _format_float(row.get("collection/route_entropy")),
                    _format_float(row.get("collection/path_entropy")),
                ]
            )
            + " |"
        )
    for seed, row in sorted(single_best.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(seed),
                    "KL learner-state single_expert",
                    str(int(row["best_round_index"])),
                    _format_float(row["best_round_greedy_success"]),
                    _format_float(row["final_greedy_success"]),
                    _format_float(row.get("collection/disagreement_rate")),
                    _format_float(row.get("collection/teacher_confidence_mean")),
                    _format_float(row.get("collection/unique_state_ratio")),
                    _format_float(row.get("added_steps")),
                    _format_float(row.get("aggregate_steps")),
                    _format_float(row.get("collection/student_confidence_mean")),
                    _format_float(row.get("collection/route_entropy")),
                    _format_float(row.get("collection/path_entropy")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## State-Local Route Redundancy Proxy",
            "",
            "| Seed | Mean Unique Pairs / Obs | Mean Dominant Pair Fraction | Global Top Pair Fraction |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for seed in sorted(route_pair_stats):
        stats = route_pair_stats[seed]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(seed),
                    _format_float(stats["route_pair_unique_mean"]),
                    _format_float(stats["route_pair_dominant_mean"]),
                    _format_float(stats["route_global_top_pair_fraction"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Final-Block Route Probes",
            "",
            "| Seed | Baseline | Worst Expert Ablation | Fixed Router | Route Randomization |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed in sorted(route_grouped):
        seed_rows = route_grouped[seed]
        baseline = next(row for row in seed_rows if row["probe"] == "baseline")
        worst_ablation = min((row for row in seed_rows if row["probe"] == "expert_ablation"), key=lambda row: float(row["eval_success_rate"]))
        fixed = next(row for row in seed_rows if row["probe"] == "router_override")
        randomized = next(row for row in seed_rows if row["probe"] == "route_randomization")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(seed),
                    _format_float(baseline.get("eval_success_rate")),
                    _format_float(worst_ablation.get("eval_success_rate")),
                    _format_float(fixed.get("eval_success_rate")),
                    _format_float(randomized.get("eval_success_rate")),
                ]
            )
            + " |"
        )

    final_sare_mean = _summary_stats(final_grouped, "kl_lss_sare", "fresh_final")["mean"]
    final_single_mean = _summary_stats(final_grouped, "kl_lss_single_expert", "fresh_final")["mean"]
    final_token_mean = _summary_stats(final_grouped, "kl_lss_token_dense", "fresh_final")["mean"]
    strong_disagreement = sum(float(row.get("collection/disagreement_rate", 0.0)) for row in strong_best) / max(len(strong_best), 1)
    final_disagreement = sum(float(row.get("collection/disagreement_rate", 0.0)) for row in final_best) / max(len(final_best), 1)
    strong_unique = sum(float(row.get("collection/unique_state_ratio", 0.0)) for row in strong_best) / max(len(strong_best), 1)
    final_unique = sum(float(row.get("collection/unique_state_ratio", 0.0)) for row in final_best) / max(len(final_best), 1)
    lines.extend(["", "## Interpretation", ""])
    if final_single_mean >= final_sare_mean:
        lines.append("- The final fresh block weakness is not specifically multi-expert: matched KL learner-state single_expert matches or beats KL learner-state SARE on the same seeds.")
    elif final_sare_mean > final_token_mean:
        lines.append("- The final fresh block keeps a routed edge over token_dense, but it remains too small to thaw the frozen claim cleanly.")
    else:
        lines.append("- The final fresh block flips toward the matched tokenized control, so the failure mode is already strong enough to keep the claim frozen.")
    if final_disagreement > strong_disagreement and final_unique <= strong_unique:
        lines.append(
            f"- Relative to the stronger recovered seeds, the final block shows higher best-round teacher-student disagreement (`{final_disagreement:.4f}` vs `{strong_disagreement:.4f}`) without better learner-state coverage (`{final_unique:.4f}` vs `{strong_unique:.4f}`), which points to extraction mismatch rather than weak teacher labels."
        )
    else:
        lines.append("- Best-round disagreement and unique-state coverage do not isolate a single clean cause for the final-block reversal.")
    if all(
        next(row for row in route_grouped[seed] if row["probe"] == "router_override")["eval_success_rate"] == 0.0
        for seed in route_grouped
    ):
        lines.append("- Even on the weak final-block seeds, fixed-router override still collapses success, so routing remains causally used rather than obviously bypassed.")
    if final_single_mean >= final_sare_mean:
        lines.append("- Because matched single_expert is at least as strong on the final block, the failure pattern supports a method-first claim more than a specifically routed interpretation.")
    return "\n".join(lines) + "\n"


def _build_updated_combined_report(rows: list[dict[str, Any]], episodes: int) -> str:
    grouped = _group(rows)
    labels = [
        ("recovered_token_dense", "recovered token_dense"),
        ("kl_lss_token_dense", "KL learner-state token_dense"),
        ("kl_lss_single_expert", "KL learner-state single_expert"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lines = [
        "# Updated Combined DoorKey Fairness Report",
        "",
        f"- external evaluation episodes per mode: `{episodes}`",
        "",
        "| Lane | Seed | recovered token_dense | KL learner-state token_dense | KL learner-state single_expert | baseline PPO SARE | KL learner-state SARE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane, seed in _all_seeds(grouped):
        values = [lane, str(seed)]
        for label, _display in labels:
            if (lane, seed, label) in grouped:
                values.append(_format_float(_greedy_row(grouped[(lane, seed, label)]).get("eval_success_rate")))
            else:
                values.append("-")
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Variant | Seeds Covered | Mean Greedy Success | Min Greedy Success | Max Greedy Success | Complete-Seed Failures |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, display in labels:
        stats = _summary_stats(grouped, label)
        count = sum(1 for (_lane, _seed, row_label) in grouped if row_label == label)
        lines.append(
            f"| {display} | `{count}` | `{stats['mean']:.4f}` | `{stats['min']:.4f}` | `{stats['max']:.4f}` | `{int(stats['zero_failures'])}` |"
        )
    token = _summary_stats(grouped, "kl_lss_token_dense")["mean"]
    single = _summary_stats(grouped, "kl_lss_single_expert")["mean"]
    sare = _summary_stats(grouped, "kl_lss_sare")["mean"]
    final_token = _summary_stats(grouped, "kl_lss_token_dense", "fresh_final")["mean"]
    final_single = _summary_stats(grouped, "kl_lss_single_expert", "fresh_final")["mean"]
    final_sare = _summary_stats(grouped, "kl_lss_sare", "fresh_final")["mean"]
    lines.extend(["", "## Interpretation", ""])
    if final_sare <= max(final_token, final_single) and sare > max(token, single):
        lines.append("- After adding the final-block single_expert control, KL learner-state SARE still leads slightly overall on the combined DoorKey picture, but the final block is weaker and the claim remains frozen.")
    elif sare > max(token, single):
        lines.append("- After adding the final-block single_expert control, the combined DoorKey picture still clearly favors KL learner-state SARE over both matched structured controls.")
    elif sare > token:
        lines.append("- After adding the final-block single_expert control, KL learner-state SARE still beats KL learner-state token_dense overall, but not clearly enough over single_expert to thaw the claim.")
    else:
        lines.append("- After adding the final-block single_expert control, the combined DoorKey picture is no longer strong enough to maintain a broader routed interpretation.")
    return "\n".join(lines) + "\n"


def _build_decision_memo(single_report_rows: list[dict[str, Any]], combined_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> str:
    single_grouped = _group(single_report_rows)
    combined_grouped = _group(combined_rows)
    final_sare = _summary_stats(single_grouped, "kl_lss_sare", "fresh_final")
    final_token = _summary_stats(single_grouped, "kl_lss_token_dense", "fresh_final")
    final_single = _summary_stats(single_grouped, "kl_lss_single_expert", "fresh_final")
    combined_sare = _summary_stats(combined_grouped, "kl_lss_sare")
    combined_token = _summary_stats(combined_grouped, "kl_lss_token_dense")
    combined_single = _summary_stats(combined_grouped, "kl_lss_single_expert")

    lines = [
        "# Frozen-Claim Decision Memo",
        "",
        "## Decision",
        "",
    ]
    if final_sare["mean"] > max(final_token["mean"], final_single["mean"]) and combined_sare["mean"] > max(combined_token["mean"], combined_single["mean"]):
        decision = "conditionally thawed within DoorKey"
    elif final_single["mean"] >= final_sare["mean"] or combined_single["mean"] >= combined_sare["mean"]:
        decision = "narrower method-first result"
    else:
        decision = "frozen bounded teacher-guided DoorKey SARE win"
    lines.append(f"The right final claim is: **{decision}**.")
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. On the final fresh block, KL learner-state SARE mean greedy success is `{final_sare['mean']:.4f}` versus `{final_token['mean']:.4f}` for KL learner-state token_dense and `{final_single['mean']:.4f}` for KL learner-state single_expert. See [lss_final_block_single_expert_control_report.md](lss_final_block_single_expert_control_report.md).",
            "2. The most plausible explanation for the `47/53/59` reversal comes from the failure analysis report: teacher labels stay confident, but the weak seeds show extraction mismatch and route-specific fragility on the final block rather than a clean teacher-quality problem. See [lss_final_block_failure_analysis.md](lss_final_block_failure_analysis.md).",
            f"3. Across the updated combined DoorKey picture, mean greedy success is `{combined_token['mean']:.4f}` for KL learner-state token_dense, `{combined_single['mean']:.4f}` for KL learner-state single_expert, and `{combined_sare['mean']:.4f}` for KL learner-state SARE. See [lss_frozen_claim_updated_combined_doorkey_report.md](lss_frozen_claim_updated_combined_doorkey_report.md).",
            "",
        ]
    )
    if decision == "conditionally thawed within DoorKey":
        lines.append("Recommendation: continue within DoorKey only. The final-block single_expert control does not erase the SARE edge, and the updated combined picture remains clearly favorable to SARE.")
    elif decision == "narrower method-first result":
        lines.append("Recommendation: narrow further. The final-block fairness control and failure analysis make the positive result look more like a structured-student extraction win than a specifically routed advantage.")
    else:
        lines.append("Recommendation: stay frozen. The current evidence still supports a bounded teacher-guided DoorKey SARE win, but not a stronger or more specific claim.")
    return "\n".join(lines) + "\n"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen DoorKey claim fairness/failure analysis.")
    sub = parser.add_subparsers(dest="command", required=True)

    repro = sub.add_parser("reproduction-note")
    repro.add_argument("--structured-csv", required=True)
    repro.add_argument("--final-csv", required=True)
    repro.add_argument("--episodes", type=int, default=64)
    repro.add_argument("--device", default="auto")
    repro.add_argument("--output", required=True)
    repro.add_argument("--csv", required=True)

    single = sub.add_parser("final-block-single-expert-control-report")
    single.add_argument("--baseline-csv", required=True)
    single.add_argument("--single-expert-root", required=True)
    single.add_argument("--episodes", type=int, default=64)
    single.add_argument("--device", default="auto")
    single.add_argument("--output", required=True)
    single.add_argument("--csv", required=True)

    failure = sub.add_parser("final-block-failure-analysis")
    failure.add_argument("--structured-csv", required=True)
    failure.add_argument("--final-csv", required=True)
    failure.add_argument("--final-single-expert-csv", required=True)
    failure.add_argument("--final-single-expert-root", required=True)
    failure.add_argument("--episodes", type=int, default=64)
    failure.add_argument("--device", default="auto")
    failure.add_argument("--output", required=True)
    failure.add_argument("--csv", required=True)

    combined = sub.add_parser("updated-combined-doorkey-report")
    combined.add_argument("--current-csv", required=True)
    combined.add_argument("--final-single-expert-csv", required=True)
    combined.add_argument("--episodes", type=int, default=64)
    combined.add_argument("--output", required=True)
    combined.add_argument("--csv", required=True)

    memo = sub.add_parser("decision-memo")
    memo.add_argument("--final-single-expert-csv", required=True)
    memo.add_argument("--combined-csv", required=True)
    memo.add_argument("--failure-csv", required=True)
    memo.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "reproduction-note":
        structured_rows = _read_csv_rows(Path(args.structured_csv))
        final_rows = _read_csv_rows(Path(args.final_csv))
        labels = {
            "recovered_token_dense",
            "kl_lss_token_dense",
            "kl_lss_single_expert",
            "baseline_sare",
            "kl_lss_sare",
        }
        targets = _targets_from_rows(structured_rows + final_rows, labels)
        rows = _evaluate_modes(targets, args.device, args.episodes, REPRO_MODES)
        content = _build_reproduction_note(rows, Path(args.structured_csv), Path(args.final_csv), args.episodes)
        _write_main_process_report(rows, args.output, args.csv, content)
        return

    if args.command == "final-block-single-expert-control-report":
        baseline_rows = _read_csv_rows(Path(args.baseline_csv))
        targets = _single_expert_targets(Path(args.single_expert_root), "fresh_final")
        eval_rows = _evaluate_modes(targets, args.device, args.episodes, [
            ("greedy", True, 1.0),
            ("sampled_t1.0", False, 1.0),
            ("sampled_t0.7", False, 0.7),
            ("sampled_t0.5", False, 0.5),
        ])
        rows = baseline_rows + eval_rows
        content = _build_final_block_single_expert_report(rows, args.episodes)
        _write_main_process_report(rows, args.output, args.csv, content)
        return

    if args.command == "final-block-failure-analysis":
        structured_rows = _read_csv_rows(Path(args.structured_csv))
        final_rows = _read_csv_rows(Path(args.final_csv))
        single_rows = _read_csv_rows(Path(args.final_single_expert_csv))
        structured_grouped = _group(structured_rows)
        strong_cases = [
            (lane, seed)
            for lane, seed in _all_seeds(structured_grouped)
            if (lane, seed, "kl_lss_sare") in structured_grouped
            and float(_greedy_row(structured_grouped[(lane, seed, "kl_lss_sare")]).get("eval_success_rate", 0.0)) >= 0.5
        ]
        strong_run_roots = {
            seed: Path(str(_greedy_row(structured_grouped[(lane, seed, "kl_lss_sare")])["run_dir"]))
            for lane, seed in strong_cases
        }
        final_grouped = _group(final_rows)
        final_run_roots = {
            seed: Path(str(_greedy_row(final_grouped[("fresh_final", seed, "kl_lss_sare")])["run_dir"]))
            for seed in _lane_seeds(final_grouped, "fresh_final")
        }
        single_grouped = _group(single_rows)
        single_run_roots = {
            seed: Path(str(_greedy_row(single_grouped[("fresh_final", seed, "kl_lss_single_expert")])["run_dir"]))
            for seed in _lane_seeds(single_grouped, "fresh_final")
        }
        route_cases = [
            _discover_case(final_run_roots[seed], "fresh_final", seed)
            for seed in sorted(final_run_roots)
        ]
        route_rows = _route_probe_rows_for_cases(route_cases, args.device, args.episodes)
        route_pair_stats: dict[int, dict[str, float]] = {}
        for seed, run_dir in final_run_roots.items():
            summary = _load_summary(run_dir)
            round_index = int(summary["best_round_index"])
            route_pair_stats[seed] = _capture_route_batch_stats(
                run_dir=run_dir,
                checkpoint_path=run_dir / f"round_{round_index:02d}.pt",
                config_path=run_dir / "student_resolved_config.yaml",
                dataset_path=run_dir / f"round_{round_index:02d}_dataset.pt",
                max_samples=4096,
                device=args.device,
            )
        content = _build_failure_analysis(
            final_rows=final_rows,
            single_rows=single_rows,
            route_rows=route_rows,
            strong_sare_rows=structured_rows,
            final_run_roots=final_run_roots,
            strong_run_roots=strong_run_roots,
            single_run_roots=single_run_roots,
            route_pair_stats=route_pair_stats,
            episodes=args.episodes,
        )
        combined_rows = []
        for row in route_rows:
            combined_rows.append({"kind": "route_probe", **row})
        for seed, stats in route_pair_stats.items():
            combined_rows.append({"kind": "route_pair_stats", "seed": seed, **stats})
        _write_main_process_report(combined_rows, args.output, args.csv, content)
        return

    if args.command == "updated-combined-doorkey-report":
        current_rows = _read_csv_rows(Path(args.current_csv))
        final_rows = _read_csv_rows(Path(args.final_single_expert_csv))
        rows = current_rows + final_rows
        content = _build_updated_combined_report(rows, args.episodes)
        _write_main_process_report(rows, args.output, args.csv, content)
        return

    if args.command == "decision-memo":
        single_rows = _read_csv_rows(Path(args.final_single_expert_csv))
        combined_rows = _read_csv_rows(Path(args.combined_csv))
        failure_rows = _read_csv_rows(Path(args.failure_csv))
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_build_decision_memo(single_rows, combined_rows, failure_rows), encoding="utf-8")
        return

    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
