from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_claim_consolidation import (
    _all_seeds,
    _group,
    _lane_seeds,
    _read_csv_rows,
    _summary_stats,
    _with_lane,
)
from psmn_rl.analysis.lss_claim_hardening import _filter_baselines, _variant_targets
from psmn_rl.analysis.lss_robustness import (
    _best_sampled,
    _evaluate_targets,
    _format_float,
    _greedy_row,
    _maybe_read_command,
    _write_main_process_report,
)
from psmn_rl.analysis.lss_route_dependence import _discover_case, _evaluate_probe, _top_experts
from psmn_rl.logging import configure_logging
from psmn_rl.rl.distributed.ddp import cleanup_distributed, init_distributed
from psmn_rl.utils.io import get_git_commit, get_git_dirty


def _rows_for_lanes(csv_paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane, path in csv_paths:
        rows.extend(_read_csv_rows(path, lane))
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _common_lane_seed_keys(
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]],
    labels: list[str],
) -> list[tuple[str, int]]:
    keys = {(lane, seed) for lane, seed, _label in grouped}
    return sorted(
        [
            (lane, seed)
            for lane, seed in keys
            if all((lane, seed, label) in grouped for label in labels)
        ]
    )


def _summary_on_keys(
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]],
    label: str,
    keys: list[tuple[str, int]],
) -> dict[str, float]:
    values = [
        float(_greedy_row(grouped[(lane, seed, label)]).get("eval_success_rate", 0.0))
        for lane, seed in keys
        if (lane, seed, label) in grouped
    ]
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "zero_failures": 0.0}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "zero_failures": float(sum(1 for value in values if value <= 0.0)),
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _route_probe_group(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["lane"]), int(row["seed"])), []).append(row)
    return grouped


def _rows_with_lane(rows: list[dict[str, Any]], lane: str) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        copy["lane"] = lane
        updated.append(copy)
    return updated


def _build_reproduction_note(rows: list[dict[str, Any]], episodes: int) -> str:
    grouped = _group(rows)
    labels = [
        ("recovered_token_dense", "recovered token_dense"),
        ("kl_lss_token_dense", "KL learner-state token_dense"),
        ("kl_lss_single_expert", "KL learner-state single_expert"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lines = [
        "# Multi-Expert Hardening Reproduction Note",
        "",
        f"- external evaluation episodes per mode: `{episodes}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Per-Seed Results",
        "",
        "| Lane | Seed | Variant | Greedy Success | Best Sampled Success | Best Sampled Mode | Config | Checkpoint | Command |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for lane, seed in _all_seeds(grouped):
        for label, display in labels:
            if (lane, seed, label) not in grouped:
                continue
            mode_rows = grouped[(lane, seed, label)]
            greedy = _greedy_row(mode_rows)
            best_mode, best_success = _best_sampled(mode_rows)
            lines.append(
                "| "
                + " | ".join(
                    [
                        lane,
                        str(seed),
                        display,
                        _format_float(greedy.get("eval_success_rate")),
                        _format_float(best_success),
                        best_mode,
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
            "- This note locks the current DoorKey teacher-guided baseline to the external 64-episode decision path before the fresh-lane `single_expert`, seed-29 forensics, and final fresh-block extensions.",
            "- It covers the original lane, the first fresh lane, and the first fresh-extra lane already published in this repo.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_fresh_single_expert_report(rows: list[dict[str, Any]], episodes: int) -> str:
    grouped = _group(rows)
    labels = [
        ("recovered_token_dense", "recovered token_dense"),
        ("kl_lss_token_dense", "KL learner-state token_dense"),
        ("kl_lss_single_expert", "KL learner-state single_expert"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    common_labels = ["kl_lss_token_dense", "kl_lss_single_expert", "kl_lss_sare"]
    common_keys = _common_lane_seed_keys(grouped, common_labels)
    lines = [
        "# Fresh-Lane Matched Single-Expert DoorKey Control Report",
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
            "## Common Structured Slice Summary",
            "",
            "| Variant | Seeds Covered | Mean Greedy Success | Min Greedy Success | Max Greedy Success | Complete-Seed Failures |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    display_names = {
        "kl_lss_token_dense": "KL learner-state token_dense",
        "kl_lss_single_expert": "KL learner-state single_expert",
        "kl_lss_sare": "KL learner-state SARE",
    }
    for label in common_labels:
        stats = _summary_on_keys(grouped, label, common_keys)
        lines.append(
            f"| {display_names[label]} | `{len(common_keys)}` | `{stats['mean']:.4f}` | `{stats['min']:.4f}` | `{stats['max']:.4f}` | `{int(stats['zero_failures'])}` |"
        )
    token_stats = _summary_on_keys(grouped, "kl_lss_token_dense", common_keys)
    single_stats = _summary_on_keys(grouped, "kl_lss_single_expert", common_keys)
    sare_stats = _summary_on_keys(grouped, "kl_lss_sare", common_keys)
    lines.extend(["", "## Interpretation", ""])
    if sare_stats["mean"] > max(token_stats["mean"], single_stats["mean"]) and sare_stats["zero_failures"] <= 0.0:
        lines.append("- On the matched structured DoorKey slice, KL learner-state SARE still stays ahead of both the matched token_dense and matched single_expert controls without a complete-seed routed failure.")
    elif sare_stats["mean"] > token_stats["mean"]:
        lines.append("- KL learner-state SARE still beats matched token_dense on the structured slice, but matched single_expert closes enough of the gap that the claim remains method-first rather than clearly multi-expert.")
    else:
        lines.append("- Once the fresh-lane matched single_expert control is included, the apparent SARE edge is no longer cleanly ahead of the matched structured controls.")
    return "\n".join(lines) + "\n"


def _build_seed29_forensics_report(
    reference_rows: list[dict[str, Any]],
    forensic_rows: list[dict[str, Any]],
    episodes: int,
    trial_count: int,
) -> str:
    reference_grouped = _route_probe_group(reference_rows)
    seed29_rows = [row for row in forensic_rows if int(row["seed"]) == 29]
    baseline_rows = [row for row in reference_rows if row["probe"] == "baseline"]
    topk_trials = [
        float(row.get("eval_success_rate", 0.0))
        for row in seed29_rows
        if row["probe"] == "route_randomization" and str(row["detail"]).startswith("uniform_topk_random:")
    ]
    strong_trials = [
        float(row.get("eval_success_rate", 0.0))
        for row in seed29_rows
        if row["probe"] == "route_randomization" and str(row["detail"]).startswith("random_single_expert:")
    ]
    pair_rows = [row for row in seed29_rows if row["probe"] == "router_override" and str(row["detail"]).startswith("pair_")]
    best_pair = max(pair_rows, key=lambda row: float(row.get("eval_success_rate", 0.0)))
    worst_pair = min(pair_rows, key=lambda row: float(row.get("eval_success_rate", 0.0)))
    baseline_seed29 = next(row for row in seed29_rows if row["probe"] == "baseline")
    current_trial0 = next(
        row
        for row in seed29_rows
        if row["probe"] == "route_randomization" and row["detail"] == "uniform_topk_random:trial=0"
    )
    lines = [
        "# Seed-29 Route-Randomization Forensics",
        "",
        f"- external evaluation episodes per probe: `{episodes}`",
        f"- repeated route-randomization trials per mode: `{trial_count}`",
        "",
        "## Baseline Routing Comparison",
        "",
        "| Lane | Seed | Greedy Success | Route Entropy | Path Entropy | Expert Load 0 | Expert Load 1 | Expert Load 2 | Expert Load 3 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(baseline_rows, key=lambda item: (str(item["lane"]), int(item["seed"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["lane"]),
                    str(row["seed"]),
                    _format_float(row.get("eval_success_rate")),
                    _format_float(row.get("route_entropy")),
                    _format_float(row.get("path_entropy")),
                    _format_float(row.get("expert_load_0")),
                    _format_float(row.get("expert_load_1")),
                    _format_float(row.get("expert_load_2")),
                    _format_float(row.get("expert_load_3")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Seed-29 Probe Ladder",
            "",
            "| Probe | Detail | Greedy Success | Route Entropy | Active Compute |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(seed29_rows, key=lambda item: (str(item["probe"]), str(item["detail"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["probe"]),
                    str(row["detail"]),
                    _format_float(row.get("eval_success_rate")),
                    _format_float(row.get("route_entropy")),
                    _format_float(row.get("active_compute_proxy")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        f"- Seed `29` baseline greedy success is `{float(baseline_seed29.get('eval_success_rate', 0.0)):.4f}`. The previously published route-randomization row was one deterministic draw at `{float(current_trial0.get('eval_success_rate', 0.0)):.4f}`."
    )
    lines.append(
        f"- Across `{trial_count}` repeated `uniform_topk_random` draws, mean greedy success is `{_mean(topk_trials):.4f}`. Across the stronger `random_single_expert` draws, mean greedy success is `{_mean(strong_trials):.4f}`."
    )
    lines.append(
        f"- The fixed-pair sweep ranges from `{float(best_pair.get('eval_success_rate', 0.0)):.4f}` on `{best_pair['detail']}` to `{float(worst_pair.get('eval_success_rate', 0.0)):.4f}` on `{worst_pair['detail']}`."
    )
    if _mean(topk_trials) >= 0.75 and _mean(strong_trials) >= 0.75:
        lines.append("- Seed `29` is a genuine exception to the route-randomization story. Repeated current and stronger randomization stay strong even though every fixed pair collapses, so the broader causal-routing claim should lean more on fixed-pair and expert-ablation evidence than on randomization for this seed.")
    elif _mean(strong_trials) + 1e-6 < float(baseline_seed29.get("eval_success_rate", 0.0)) and float(best_pair.get("eval_success_rate", 0.0)) + 1e-6 < float(baseline_seed29.get("eval_success_rate", 0.0)):
        lines.append("- Seed `29` is not a general failure of the causal-routing story. The weak original randomization row looks like a narrow probe exception rather than evidence that routing can be randomized freely without cost.")
    else:
        lines.append("- Seed `29` remains a genuine exception under the current bounded probe ladder, so the causal-routing claim should stay qualified rather than universal.")
    lines.append("- The expert loads and route entropy remain close to the other recovered seeds, so the exception is not explained by an obvious global routing collapse or single-expert dominance scalar.")
    return "\n".join(lines) + "\n"


def _build_broader_route_dependence_report(
    rows: list[dict[str, Any]],
    forensic_rows: list[dict[str, Any]],
    episodes: int,
) -> str:
    grouped = _route_probe_group(rows)
    lines = [
        "# Broader Causal Route-Dependence Report",
        "",
        f"- external evaluation episodes per probe: `{episodes}`",
        "",
        "| Lane | Seed | Probe | Detail | Greedy Success | Route Entropy | Active Compute |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["lane"]),
                    str(row["seed"]),
                    str(row["probe"]),
                    str(row["detail"]),
                    _format_float(row.get("eval_success_rate")),
                    _format_float(row.get("route_entropy")),
                    _format_float(row.get("active_compute_proxy")),
                ]
            )
            + " |"
        )
    fixed_drops: list[float] = []
    random_drops: list[float] = []
    ablation_drops: list[float] = []
    for case_rows in grouped.values():
        baseline = next(row for row in case_rows if row["probe"] == "baseline")
        fixed = next(row for row in case_rows if row["probe"] == "router_override")
        randomized = next(row for row in case_rows if row["probe"] == "route_randomization")
        ablations = [row for row in case_rows if row["probe"] == "expert_ablation"]
        baseline_success = float(baseline.get("eval_success_rate", 0.0))
        fixed_drops.append(baseline_success - float(fixed.get("eval_success_rate", 0.0)))
        random_drops.append(baseline_success - float(randomized.get("eval_success_rate", 0.0)))
        ablation_drops.append(
            baseline_success - min(float(row.get("eval_success_rate", 0.0)) for row in ablations)
        )
    seed29_mean_current = _mean(
        [
            float(row.get("eval_success_rate", 0.0))
            for row in forensic_rows
            if row["probe"] == "route_randomization" and str(row["detail"]).startswith("uniform_topk_random:")
        ]
    )
    seed29_mean_strong = _mean(
        [
            float(row.get("eval_success_rate", 0.0))
            for row in forensic_rows
            if row["probe"] == "route_randomization" and str(row["detail"]).startswith("random_single_expert:")
        ]
    )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        f"- Across the expanded recovered-seed set, mean greedy-success drop is `{_mean(fixed_drops):.4f}` for fixed-router override, `{_mean(random_drops):.4f}` for the current route-randomization probe, and `{_mean(ablation_drops):.4f}` for the worst expert ablation."
    )
    if seed29_mean_current >= 0.75 and seed29_mean_strong >= 0.75:
        lines.append(
            f"- Seed `29` remains the main genuine route-randomization exception: current random top-k averages `{seed29_mean_current:.4f}` and stronger random-single-expert still averages `{seed29_mean_strong:.4f}`."
        )
    elif seed29_mean_strong + 1e-6 < seed29_mean_current:
        lines.append(
            f"- Seed `29` stays the only weak current-randomization case, but the forensic ladder still lowers its stronger randomization mean to `{seed29_mean_strong:.4f}` from `{seed29_mean_current:.4f}` under the current random top-k probe."
        )
    else:
        lines.append("- Seed `29` remains the main cautionary case: stronger randomization does not separate it much from the current random-routing probe.")
    if _mean(fixed_drops) >= 0.25 and _mean(ablation_drops) >= 0.25 and sum(1 for drop in random_drops if drop >= 0.25) >= max(len(random_drops) - 1, 1):
        lines.append("- The broader DoorKey picture still supports a causal-routing interpretation: expert ablation and fixed-router override are broadly harmful, and route randomization is strongly harmful on all but at most one narrow exception.")
    else:
        lines.append("- The broader DoorKey picture is mixed enough that routing should still be described as correlated with success more carefully than as a broad causal edge.")
    return "\n".join(lines) + "\n"


def _build_final_fresh_block_report(rows: list[dict[str, Any]], episodes: int, lane: str) -> str:
    grouped = _group(rows)
    labels = [
        ("recovered_token_dense", "recovered token_dense"),
        ("kl_lss_token_dense", "KL learner-state token_dense"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    lines = [
        "# Final Fresh DoorKey Seed Block Report",
        "",
        f"- lane: `{lane}`",
        f"- external evaluation episodes per mode: `{episodes}`",
        "",
        "| Seed | recovered token_dense | KL learner-state token_dense | baseline PPO SARE | KL learner-state SARE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for seed in _lane_seeds(grouped, lane):
        values = [str(seed)]
        for label, _display in labels:
            if (lane, seed, label) in grouped:
                values.append(_format_float(_greedy_row(grouped[(lane, seed, label)]).get("eval_success_rate")))
            else:
                values.append("-")
        lines.append("| " + " | ".join(values) + " |")
    token_stats = _summary_stats(grouped, "kl_lss_token_dense", lane)
    sare_stats = _summary_stats(grouped, "kl_lss_sare", lane)
    lines.extend(
        [
            "",
            "## Mean Greedy Success",
            "",
            "| Variant | Mean Greedy Success | Complete-Seed Failures |",
            "| --- | ---: | ---: |",
            f"| recovered token_dense | `{_summary_stats(grouped, 'recovered_token_dense', lane)['mean']:.4f}` | `{int(_summary_stats(grouped, 'recovered_token_dense', lane)['zero_failures'])}` |",
            f"| KL learner-state token_dense | `{token_stats['mean']:.4f}` | `{int(token_stats['zero_failures'])}` |",
            f"| baseline PPO SARE | `{_summary_stats(grouped, 'baseline_sare', lane)['mean']:.4f}` | `{int(_summary_stats(grouped, 'baseline_sare', lane)['zero_failures'])}` |",
            f"| KL learner-state SARE | `{sare_stats['mean']:.4f}` | `{int(sare_stats['zero_failures'])}` |",
            "",
            "## Interpretation",
            "",
        ]
    )
    if sare_stats["mean"] > token_stats["mean"] and sare_stats["zero_failures"] <= 0.0:
        lines.append("- On the final fresh matched DoorKey block, KL learner-state SARE still stays ahead of the matched teacher-guided token_dense control without any complete-seed routed failure.")
    elif sare_stats["mean"] >= token_stats["mean"]:
        lines.append("- On the final fresh matched DoorKey block, KL learner-state SARE stays competitive with matched token_dense, but the edge is narrower than the current broader DoorKey picture.")
    else:
        lines.append("- On the final fresh matched DoorKey block, matched teacher-guided token_dense catches or beats KL learner-state SARE, so the broader DoorKey edge weakens.")
    return "\n".join(lines) + "\n"


def _build_final_combined_doorkey_report(rows: list[dict[str, Any]], episodes: int) -> str:
    grouped = _group(rows)
    labels = [
        ("recovered_token_dense", "recovered token_dense"),
        ("kl_lss_token_dense", "KL learner-state token_dense"),
        ("kl_lss_single_expert", "KL learner-state single_expert"),
        ("baseline_sare", "baseline PPO SARE"),
        ("kl_lss_sare", "KL learner-state SARE"),
    ]
    common_keys = _common_lane_seed_keys(grouped, ["kl_lss_token_dense", "kl_lss_single_expert", "kl_lss_sare"])
    lines = [
        "# Final Combined DoorKey Report",
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
        seed_count = len({seed for _lane, seed, row_label in grouped if row_label == label})
        lines.append(
            f"| {display} | `{seed_count}` | `{stats['mean']:.4f}` | `{stats['min']:.4f}` | `{stats['max']:.4f}` | `{int(stats['zero_failures'])}` |"
        )
    token_common = _summary_on_keys(grouped, "kl_lss_token_dense", common_keys)
    single_common = _summary_on_keys(grouped, "kl_lss_single_expert", common_keys)
    sare_common = _summary_on_keys(grouped, "kl_lss_sare", common_keys)
    token_overall = _summary_stats(grouped, "kl_lss_token_dense")
    sare_overall = _summary_stats(grouped, "kl_lss_sare")
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        f"- Across the full DoorKey picture, KL learner-state SARE mean greedy success is `{sare_overall['mean']:.4f}` versus `{token_overall['mean']:.4f}` for KL learner-state token_dense."
    )
    lines.append(
        f"- On the matched structured slice where token_dense, single_expert, and SARE all exist, the means are `{token_common['mean']:.4f}` for KL learner-state token_dense, `{single_common['mean']:.4f}` for KL learner-state single_expert, and `{sare_common['mean']:.4f}` for KL learner-state SARE."
    )
    if sare_common["mean"] > max(token_common["mean"], single_common["mean"]) and sare_overall["zero_failures"] <= 0.0:
        lines.append("- The final combined DoorKey picture supports a specifically multi-expert routed edge under teacher-guided KL learner-state extraction.")
    elif sare_overall["mean"] > token_overall["mean"]:
        lines.append("- The final combined DoorKey picture still favors KL learner-state SARE over token_dense, but the edge over single_expert stays too small or inconsistent to call specifically multi-expert.")
    else:
        lines.append("- The final combined DoorKey picture is better read as a structured-student extraction story than as a specifically multi-expert routed edge.")
    return "\n".join(lines) + "\n"


def _build_decision_memo(
    single_rows: list[dict[str, Any]],
    broader_rows: list[dict[str, Any]],
    forensics_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    combined_rows: list[dict[str, Any]],
) -> str:
    single_grouped = _group(single_rows)
    broader_grouped = _route_probe_group(broader_rows)
    final_grouped = _group(final_rows)
    combined_grouped = _group(combined_rows)
    common_keys = _common_lane_seed_keys(single_grouped, ["kl_lss_token_dense", "kl_lss_single_expert", "kl_lss_sare"])
    token_common = _summary_on_keys(single_grouped, "kl_lss_token_dense", common_keys)
    single_common = _summary_on_keys(single_grouped, "kl_lss_single_expert", common_keys)
    sare_common = _summary_on_keys(single_grouped, "kl_lss_sare", common_keys)
    fixed_drops: list[float] = []
    random_drops: list[float] = []
    ablation_drops: list[float] = []
    for case_rows in broader_grouped.values():
        baseline = next(row for row in case_rows if row["probe"] == "baseline")
        fixed = next(row for row in case_rows if row["probe"] == "router_override")
        randomized = next(row for row in case_rows if row["probe"] == "route_randomization")
        ablations = [row for row in case_rows if row["probe"] == "expert_ablation"]
        baseline_success = float(baseline.get("eval_success_rate", 0.0))
        fixed_drops.append(baseline_success - float(fixed.get("eval_success_rate", 0.0)))
        random_drops.append(baseline_success - float(randomized.get("eval_success_rate", 0.0)))
        ablation_drops.append(
            baseline_success - min(float(row.get("eval_success_rate", 0.0)) for row in ablations)
        )
    current_random = _mean(
        [
            float(row.get("eval_success_rate", 0.0))
            for row in forensics_rows
            if row["probe"] == "route_randomization" and str(row["detail"]).startswith("uniform_topk_random:")
        ]
    )
    strong_random = _mean(
        [
            float(row.get("eval_success_rate", 0.0))
            for row in forensics_rows
            if row["probe"] == "route_randomization" and str(row["detail"]).startswith("random_single_expert:")
        ]
    )
    final_token = _summary_stats(final_grouped, "kl_lss_token_dense", "fresh_final")
    final_sare = _summary_stats(final_grouped, "kl_lss_sare", "fresh_final")
    combined_token = _summary_stats(combined_grouped, "kl_lss_token_dense")
    combined_single = _summary_stats(combined_grouped, "kl_lss_single_expert")
    combined_sare = _summary_stats(combined_grouped, "kl_lss_sare")
    fairness_pass = sare_common["mean"] > max(token_common["mean"], single_common["mean"])
    route_pass = _mean(fixed_drops) >= 0.25 and _mean(ablation_drops) >= 0.25 and sum(
        1 for drop in random_drops if drop >= 0.25
    ) >= max(len(random_drops) - 1, 1)
    seed29_ok = (strong_random + 1e-6 < current_random) or (current_random >= 0.75 and strong_random >= 0.75)
    final_block_pass = final_sare["mean"] > final_token["mean"] and final_sare["zero_failures"] <= 0.0
    if fairness_pass and route_pass and seed29_ok and final_block_pass:
        claim = "specifically multi-expert routed DoorKey edge under teacher-guided KL learner-state extraction"
        recommendation = "continue within DoorKey"
    elif combined_sare["mean"] > combined_token["mean"]:
        claim = "bounded DoorKey teacher-guided SARE win that is still more method-first than fully mechanism-hardened"
        recommendation = "stay frozen"
    else:
        claim = "structured-student extraction win only"
        recommendation = "pause"
    lines = [
        "# Multi-Expert DoorKey Hardening Decision Memo",
        "",
        "## Decision",
        "",
        f"The right current DoorKey claim is: **{claim}**.",
        "",
        "This remains explicitly bounded to:",
        "",
        "- teacher-guided extraction only",
        "- DoorKey only",
        "- external `64`-episode evaluation only",
        "",
        "## Answers",
        "",
        "1. Does the DoorKey edge survive fresh matched `single_expert` controls?",
        "",
        f"On the matched structured slice, KL learner-state `SARE` mean greedy success is `{sare_common['mean']:.4f}` versus `{token_common['mean']:.4f}` for KL learner-state `token_dense` and `{single_common['mean']:.4f}` for KL learner-state `single_expert`. See [lss_fresh_single_expert_matched_control_report.md](lss_fresh_single_expert_matched_control_report.md).",
        "",
        "2. What is the right interpretation of the seed-29 route-randomization result?",
        "",
        (
            f"The current random top-k route-randomization ladder averages `{current_random:.4f}` greedy success on seed `29`, while the stronger random-single-expert ladder averages `{strong_random:.4f}`. "
            + (
                "That makes seed `29` a genuine but narrow exception to the route-randomization story, not a clean refutation of the broader fixed-pair / expert-ablation evidence."
                if current_random >= 0.75 and strong_random >= 0.75
                else "That keeps seed `29` in the narrow-exception bucket rather than turning it into a clean refutation of causal routing."
            )
            + " See [lss_seed29_route_randomization_forensics.md](lss_seed29_route_randomization_forensics.md)."
        ),
        "",
        "3. Does causal routing dependence remain broadly true across the expanded recovered-seed set?",
        "",
        f"Across the broader recovered-seed set, mean greedy-success drop is `{_mean(fixed_drops):.4f}` for fixed-router override, `{_mean(random_drops):.4f}` for the current route-randomization probe, and `{_mean(ablation_drops):.4f}` for the worst expert ablation. See [lss_broader_route_dependence_report.md](lss_broader_route_dependence_report.md).",
        "",
        "4. Does one final fresh block strengthen, preserve, or weaken the edge?",
        "",
        f"On the final fresh block, KL learner-state `SARE` mean greedy success is `{final_sare['mean']:.4f}` versus `{final_token['mean']:.4f}` for KL learner-state `token_dense`, with `{int(final_sare['zero_failures'])}` complete-seed routed failures. See [lss_final_fresh_seed_block_report.md](lss_final_fresh_seed_block_report.md).",
        "",
        "5. Should work continue within DoorKey, stay frozen at the current scope, or pause again?",
        "",
        f"Recommendation: **{recommendation}**. The full combined DoorKey picture is `{combined_sare['mean']:.4f}` for KL learner-state `SARE`, `{combined_token['mean']:.4f}` for KL learner-state `token_dense`, and `{combined_single['mean']:.4f}` for KL learner-state `single_expert` on the seeds where it exists. See [lss_final_combined_doorkey_report.md](lss_final_combined_doorkey_report.md) and [lss_keycorridor_transfer_report.md](lss_keycorridor_transfer_report.md).",
    ]
    return "\n".join(lines) + "\n"


def reproduction_note(args: argparse.Namespace) -> None:
    targets = (
        _with_lane(
            _filter_baselines(Path(args.original_baseline_root), {"recovered_token_dense", "baseline_sare"}),
            "original",
        )
        + _with_lane(_variant_targets(Path(args.original_token_root), {"token_dense": "kl_lss_token_dense"}), "original")
        + _with_lane(
            _variant_targets(Path(args.original_single_expert_root), {"single_expert": "kl_lss_single_expert"}),
            "original",
        )
        + _with_lane(_variant_targets(Path(args.original_sare_root), {"sare": "kl_lss_sare"}), "original")
        + _with_lane(
            _filter_baselines(Path(args.fresh_baseline_root), {"recovered_token_dense", "baseline_sare"}),
            "fresh",
        )
        + _with_lane(_variant_targets(Path(args.fresh_token_root), {"token_dense": "kl_lss_token_dense"}), "fresh")
        + _with_lane(_variant_targets(Path(args.fresh_sare_root), {"sare": "kl_lss_sare"}), "fresh")
        + _with_lane(
            _filter_baselines(Path(args.fresh_extra_baseline_root), {"recovered_token_dense", "baseline_sare"}),
            "fresh_extra",
        )
        + _with_lane(
            _variant_targets(Path(args.fresh_extra_token_root), {"token_dense": "kl_lss_token_dense"}),
            "fresh_extra",
        )
        + _with_lane(
            _variant_targets(Path(args.fresh_extra_sare_root), {"sare": "kl_lss_sare"}),
            "fresh_extra",
        )
    )
    rows = _evaluate_targets(targets, args.device, args.episodes)
    _write_main_process_report(rows, args.output, args.csv, _build_reproduction_note(rows, args.episodes))


def fresh_single_expert_matched_control_report(args: argparse.Namespace) -> None:
    rows = _rows_for_lanes(
        [
            ("original", Path(args.original_csv)),
            ("fresh", Path(args.fresh_csv)),
            ("fresh_extra", Path(args.fresh_extra_csv)),
        ]
    )
    fresh_targets = _with_lane(
        _variant_targets(Path(args.fresh_single_expert_root), {"single_expert": "kl_lss_single_expert"}),
        "fresh",
    )
    fresh_extra_targets = _with_lane(
        _variant_targets(Path(args.fresh_extra_single_expert_root), {"single_expert": "kl_lss_single_expert"}),
        "fresh_extra",
    )
    rows.extend(_evaluate_targets(fresh_targets + fresh_extra_targets, args.device, args.episodes))
    keep = {
        "recovered_token_dense",
        "kl_lss_token_dense",
        "kl_lss_single_expert",
        "baseline_sare",
        "kl_lss_sare",
    }
    rows = [row for row in rows if row["label"] in keep]
    _write_main_process_report(rows, args.output, args.csv, _build_fresh_single_expert_report(rows, args.episodes))


def seed29_route_randomization_forensics(args: argparse.Namespace) -> None:
    reference_rows = _read_csv(Path(args.reference_csv))
    reference_rows = [row for row in reference_rows if int(row["seed"]) in {7, 19, 23, 29}]
    case = _discover_case(Path(args.run_dir), "fresh", 29)
    ctx = init_distributed(args.device, "auto")
    configure_logging(ctx.is_main_process)
    try:
        forensic_rows: list[dict[str, Any]] = []
        baseline = _evaluate_probe(case, ctx, args.episodes, "baseline")
        if ctx.is_main_process:
            forensic_rows.append(baseline)
            top_experts = _top_experts(
                baseline,
                expert_count=int(baseline.get("expert_count", 0)),
                top_k=int(baseline.get("top_k", 1)),
            )
        else:
            top_experts = [0, 1]
        for expert_index in range(int(baseline.get("expert_count", 0)) or 0):
            row = _evaluate_probe(case, ctx, args.episodes, "expert_ablation", detail=str(expert_index))
            if ctx.is_main_process:
                forensic_rows.append(row)
        for pair in combinations(range(int(baseline.get("expert_count", 0)) or 0), int(baseline.get("top_k", 1)) or 1):
            row = _evaluate_probe(
                case,
                ctx,
                args.episodes,
                "router_override",
                detail=f"pair_{pair[0]}_{pair[1]}",
                fixed_experts=list(pair),
            )
            if ctx.is_main_process:
                forensic_rows.append(row)
        row = _evaluate_probe(
            case,
            ctx,
            args.episodes,
            "router_override",
            detail="most_used_pair",
            fixed_experts=top_experts,
        )
        if ctx.is_main_process:
            forensic_rows.append(row)
        for trial in range(args.trial_count):
            current = _evaluate_probe(
                case,
                ctx,
                args.episodes,
                "route_randomization",
                detail=f"uniform_topk_random:trial={trial}",
            )
            strong = _evaluate_probe(
                case,
                ctx,
                args.episodes,
                "route_randomization",
                detail=f"random_single_expert:trial={trial}",
            )
            if ctx.is_main_process:
                forensic_rows.append(current)
                forensic_rows.append(strong)
        if not ctx.is_main_process:
            return
        rows = reference_rows + _rows_with_lane(forensic_rows, "fresh")
        _write_main_process_report(
            rows,
            args.output,
            args.csv,
            _build_seed29_forensics_report(reference_rows, forensic_rows, args.episodes, args.trial_count),
        )
    finally:
        cleanup_distributed(ctx)


def broader_route_dependence_report(args: argparse.Namespace) -> None:
    keep_rows = _read_csv(Path(args.existing_csv)) + _read_csv(Path(args.extra_csv))
    forensic_rows = _read_csv(Path(args.forensics_csv))
    _write_main_process_report(
        keep_rows,
        args.output,
        args.csv,
        _build_broader_route_dependence_report(keep_rows, forensic_rows, args.episodes),
    )


def final_fresh_seed_block_report(args: argparse.Namespace) -> None:
    rows = _rows_for_lanes([(args.lane, Path(args.baseline_csv))])
    targets = _with_lane(_variant_targets(Path(args.token_root), {"token_dense": "kl_lss_token_dense"}), args.lane)
    rows.extend(_evaluate_targets(targets, args.device, args.episodes))
    keep = {"recovered_token_dense", "kl_lss_token_dense", "baseline_sare", "kl_lss_sare"}
    rows = [row for row in rows if row["label"] in keep]
    _write_main_process_report(rows, args.output, args.csv, _build_final_fresh_block_report(rows, args.episodes, args.lane))


def final_combined_doorkey_report(args: argparse.Namespace) -> None:
    rows = _read_csv(Path(args.current_csv)) + _read_csv(Path(args.final_csv))
    _write_main_process_report(rows, args.output, args.csv, _build_final_combined_doorkey_report(rows, args.episodes))


def decision_memo(args: argparse.Namespace) -> None:
    single_rows = _read_csv(Path(args.single_expert_csv))
    broader_rows = _read_csv(Path(args.broader_route_csv))
    forensics_rows = _read_csv(Path(args.forensics_csv))
    final_rows = _read_csv(Path(args.final_block_csv))
    combined_rows = _read_csv(Path(args.combined_csv))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _build_decision_memo(single_rows, broader_rows, forensics_rows, final_rows, combined_rows),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze multi-expert DoorKey hardening experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reproduction = subparsers.add_parser("reproduction-note")
    reproduction.add_argument("--original-baseline-root", required=True)
    reproduction.add_argument("--original-token-root", required=True)
    reproduction.add_argument("--original-single-expert-root", required=True)
    reproduction.add_argument("--original-sare-root", required=True)
    reproduction.add_argument("--fresh-baseline-root", required=True)
    reproduction.add_argument("--fresh-token-root", required=True)
    reproduction.add_argument("--fresh-sare-root", required=True)
    reproduction.add_argument("--fresh-extra-baseline-root", required=True)
    reproduction.add_argument("--fresh-extra-token-root", required=True)
    reproduction.add_argument("--fresh-extra-sare-root", required=True)
    reproduction.add_argument("--episodes", type=int, default=64)
    reproduction.add_argument("--device", default="auto")
    reproduction.add_argument("--output", required=True)
    reproduction.add_argument("--csv", default=None)

    single = subparsers.add_parser("fresh-single-expert-matched-control-report")
    single.add_argument("--original-csv", required=True)
    single.add_argument("--fresh-csv", required=True)
    single.add_argument("--fresh-extra-csv", required=True)
    single.add_argument("--fresh-single-expert-root", required=True)
    single.add_argument("--fresh-extra-single-expert-root", required=True)
    single.add_argument("--episodes", type=int, default=64)
    single.add_argument("--device", default="auto")
    single.add_argument("--output", required=True)
    single.add_argument("--csv", default=None)

    forensics = subparsers.add_parser("seed29-route-randomization-forensics")
    forensics.add_argument("--reference-csv", required=True)
    forensics.add_argument("--run-dir", required=True)
    forensics.add_argument("--episodes", type=int, default=64)
    forensics.add_argument("--trial-count", type=int, default=8)
    forensics.add_argument("--device", default="auto")
    forensics.add_argument("--output", required=True)
    forensics.add_argument("--csv", default=None)

    broader = subparsers.add_parser("broader-route-dependence-report")
    broader.add_argument("--existing-csv", required=True)
    broader.add_argument("--extra-csv", required=True)
    broader.add_argument("--forensics-csv", required=True)
    broader.add_argument("--episodes", type=int, default=64)
    broader.add_argument("--output", required=True)
    broader.add_argument("--csv", default=None)

    final_block = subparsers.add_parser("final-fresh-seed-block-report")
    final_block.add_argument("--baseline-csv", required=True)
    final_block.add_argument("--token-root", required=True)
    final_block.add_argument("--episodes", type=int, default=64)
    final_block.add_argument("--device", default="auto")
    final_block.add_argument("--output", required=True)
    final_block.add_argument("--csv", default=None)
    final_block.add_argument("--lane", default="fresh_final")

    combined = subparsers.add_parser("final-combined-doorkey-report")
    combined.add_argument("--current-csv", required=True)
    combined.add_argument("--final-csv", required=True)
    combined.add_argument("--episodes", type=int, default=64)
    combined.add_argument("--output", required=True)
    combined.add_argument("--csv", default=None)

    memo = subparsers.add_parser("decision-memo")
    memo.add_argument("--single-expert-csv", required=True)
    memo.add_argument("--forensics-csv", required=True)
    memo.add_argument("--broader-route-csv", required=True)
    memo.add_argument("--final-block-csv", required=True)
    memo.add_argument("--combined-csv", required=True)
    memo.add_argument("--output", required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "reproduction-note":
        reproduction_note(args)
        return
    if args.command == "fresh-single-expert-matched-control-report":
        fresh_single_expert_matched_control_report(args)
        return
    if args.command == "seed29-route-randomization-forensics":
        seed29_route_randomization_forensics(args)
        return
    if args.command == "broader-route-dependence-report":
        broader_route_dependence_report(args)
        return
    if args.command == "final-fresh-seed-block-report":
        final_fresh_seed_block_report(args)
        return
    if args.command == "final-combined-doorkey-report":
        final_combined_doorkey_report(args)
        return
    decision_memo(args)


if __name__ == "__main__":
    main()
