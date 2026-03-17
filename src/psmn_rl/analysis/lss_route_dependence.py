from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Iterator

import torch

from psmn_rl.analysis.lss_robustness import _format_float
from psmn_rl.config import load_config
from psmn_rl.logging import configure_logging
from psmn_rl.models.factory import build_model
from psmn_rl.models.routing.sare import RoutedExpertCore, _gather_expert_outputs
from psmn_rl.rl.distributed.ddp import cleanup_distributed, init_distributed
from psmn_rl.rl.ppo.algorithm import collect_policy_diagnostics
from psmn_rl.utils.io import get_git_commit, get_git_dirty
from psmn_rl.utils.seed import set_seed


def _discover_case(run_dir: Path, lane: str, seed: int) -> dict[str, Any]:
    return {
        "lane": lane,
        "seed": seed,
        "run_dir": run_dir,
        "config_path": run_dir / "resolved_config.yaml",
        "checkpoint_path": run_dir / "latest.pt",
    }


def _parse_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.case:
        return [
            _discover_case(Path(run_dir), str(lane), int(seed))
            for lane, seed, run_dir in args.case
        ]
    if args.original_run is None or args.fresh_run is None:
        raise ValueError("either --case must be provided or both --original-run and --fresh-run must be set")
    return [
        _discover_case(Path(args.original_run), "original", 7),
        _discover_case(Path(args.fresh_run), "fresh", 23),
    ]


def _build_model(case: dict[str, Any], device: torch.device) -> tuple[Any, Any]:
    config = load_config(case["config_path"])
    config.system.device = str(device)
    config.logging.tensorboard = False
    envs = None
    try:
        from psmn_rl.envs.registry import make_vector_env

        envs = make_vector_env(config.env, seed=config.seed, world_rank=0)
        model = build_model(config.model, envs.single_observation_space, envs.single_action_space).to(device)
    finally:
        if envs is not None:
            envs.close()
    checkpoint = torch.load(case["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return config, model


def _top_experts(metrics: dict[str, Any], expert_count: int, top_k: int) -> list[int]:
    loads = []
    for expert_index in range(expert_count):
        loads.append((expert_index, float(metrics.get(f"expert_load_{expert_index}", 0.0))))
    loads.sort(key=lambda item: item[1], reverse=True)
    selected = [expert_index for expert_index, _load in loads[:top_k]]
    if not selected:
        selected = [0]
    while len(selected) < top_k:
        selected.append(selected[-1])
    return selected


@contextmanager
def _patched_core(core: RoutedExpertCore, probe: str, detail: str | None, fixed_experts: list[int] | None) -> Iterator[None]:
    original_route = core.route
    original_apply = core.apply_experts
    try:
        if probe == "expert_ablation":
            expert_index = int(detail)

            def apply_experts(self: RoutedExpertCore, tokens: torch.Tensor, topk_values: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
                expert_outputs = self.bank.forward_all(tokens).clone()
                expert_outputs[:, :, expert_index, :] = 0.0
                gathered = _gather_expert_outputs(expert_outputs, topk_idx)
                return (gathered * topk_values.unsqueeze(-1)).sum(dim=2)

            core.apply_experts = MethodType(apply_experts, core)
        elif probe == "router_override":
            assert fixed_experts is not None

            def route(self: RoutedExpertCore, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                batch, token_count, _hidden = tokens.shape
                top_k = min(self.top_k, self.expert_count)
                chosen = fixed_experts[:top_k]
                index = torch.as_tensor(chosen, device=tokens.device, dtype=torch.long)
                topk_idx = index.view(1, 1, -1).expand(batch, token_count, -1).clone()
                route_probs = torch.zeros(batch, token_count, self.expert_count, device=tokens.device, dtype=tokens.dtype)
                route_probs.scatter_(-1, topk_idx, 1.0 / top_k)
                topk_values = torch.full((batch, token_count, top_k), 1.0 / top_k, device=tokens.device, dtype=tokens.dtype)
                return route_probs, topk_values, topk_idx

            core.route = MethodType(route, core)
        elif probe == "route_randomization":
            def route(self: RoutedExpertCore, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                batch, token_count, _hidden = tokens.shape
                top_k = min(self.top_k, self.expert_count)
                random_scores = torch.rand(batch, token_count, self.expert_count, device=tokens.device, dtype=tokens.dtype)
                topk_idx = torch.topk(random_scores, k=top_k, dim=-1).indices
                route_probs = torch.zeros(batch, token_count, self.expert_count, device=tokens.device, dtype=tokens.dtype)
                route_probs.scatter_(-1, topk_idx, 1.0 / top_k)
                topk_values = torch.full((batch, token_count, top_k), 1.0 / top_k, device=tokens.device, dtype=tokens.dtype)
                return route_probs, topk_values, topk_idx

            core.route = MethodType(route, core)
        yield
    finally:
        core.route = original_route
        core.apply_experts = original_apply


def _evaluate_probe(
    case: dict[str, Any],
    ctx,
    episodes: int,
    probe: str,
    detail: str | None = None,
    fixed_experts: list[int] | None = None,
) -> dict[str, Any]:
    config, model = _build_model(case, ctx.device)
    if not isinstance(model.core, RoutedExpertCore):
        raise ValueError(f"route dependence requires RoutedExpertCore, got {type(model.core).__name__}")
    set_seed(config.seed + ctx.rank, deterministic=config.system.deterministic)
    with _patched_core(model.core, probe, detail, fixed_experts):
        diagnostics = collect_policy_diagnostics(
            config=config,
            model=model,
            ctx=ctx,
            episodes=episodes,
            greedy=True,
            temperature=1.0,
            trace_limit=0,
        )
    return {
        "lane": case["lane"],
        "seed": case["seed"],
        "probe": probe,
        "detail": detail or "-",
        "expert_count": float(model.core.expert_count),
        "top_k": float(model.core.top_k),
        "config_path": str(case["config_path"]),
        "checkpoint_path": str(case["checkpoint_path"]),
        "run_dir": str(case["run_dir"]),
        **diagnostics.metrics,
    }


def _build_report(rows: list[dict[str, Any]], top_experts_by_case: dict[tuple[str, int], list[int]], episodes: int) -> str:
    lines = [
        "# Causal Route-Dependence Report",
        "",
        f"- external evaluation episodes per probe: `{episodes}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "| Lane | Seed | Probe | Detail | Greedy Success | Greedy Return | Route Entropy | Active Compute |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["lane"]), int(row["seed"])), []).append(row)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["lane"]),
                    str(row["seed"]),
                    str(row["probe"]),
                    str(row["detail"]),
                    _format_float(row.get("eval_success_rate")),
                    _format_float(row.get("eval_return")),
                    _format_float(row.get("route_entropy")),
                    _format_float(row.get("active_compute_proxy")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Interpretation", ""])
    for (lane, seed), case_rows in sorted(grouped.items()):
        baseline = next(row for row in case_rows if row["probe"] == "baseline")
        fixed = next(row for row in case_rows if row["probe"] == "router_override")
        randomized = next(row for row in case_rows if row["probe"] == "route_randomization")
        ablations = [row for row in case_rows if row["probe"] == "expert_ablation"]
        worst_ablation = min(ablations, key=lambda row: float(row.get("eval_success_rate", 0.0)))
        fixed_drop = float(baseline.get("eval_success_rate", 0.0)) - float(fixed.get("eval_success_rate", 0.0))
        random_drop = float(baseline.get("eval_success_rate", 0.0)) - float(randomized.get("eval_success_rate", 0.0))
        worst_drop = float(baseline.get("eval_success_rate", 0.0)) - float(worst_ablation.get("eval_success_rate", 0.0))
        lines.append(
            f"- `{lane}` seed `{seed}` uses top experts `{top_experts_by_case[(lane, seed)]}` for the fixed-router probe. "
            f"Baseline greedy success is `{baseline.get('eval_success_rate', 0.0):.4f}`; fixed-router drop is `{fixed_drop:.4f}`, "
            f"random-routing drop is `{random_drop:.4f}`, and worst single-expert ablation drop is `{worst_drop:.4f}` "
            f"(expert `{worst_ablation['detail']}`)."
        )
    fixed_mean_drop = 0.0
    random_mean_drop = 0.0
    case_count = max(len(grouped), 1)
    for case_rows in grouped.values():
        baseline = next(row for row in case_rows if row["probe"] == "baseline")
        fixed = next(row for row in case_rows if row["probe"] == "router_override")
        randomized = next(row for row in case_rows if row["probe"] == "route_randomization")
        fixed_mean_drop += float(baseline.get("eval_success_rate", 0.0)) - float(fixed.get("eval_success_rate", 0.0))
        random_mean_drop += float(baseline.get("eval_success_rate", 0.0)) - float(randomized.get("eval_success_rate", 0.0))
    fixed_mean_drop /= case_count
    random_mean_drop /= case_count
    if max(fixed_mean_drop, random_mean_drop) >= 0.25:
        lines.append("- Learned routing assignments are causally relevant under this bounded probe family: disrupting routing materially degrades greedy DoorKey success.")
    else:
        lines.append("- Learned routing assignments are not strongly causally supported by this bounded probe family: routing disruption causes only small changes in greedy DoorKey success.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal route-dependence probes for recovered SARE checkpoints.")
    parser.add_argument("--original-run", default=None)
    parser.add_argument("--fresh-run", default=None)
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        metavar=("LANE", "SEED", "RUN_DIR"),
        help="Evaluate one recovered SARE run as lane/seed/run_dir. May be repeated.",
    )
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    cases = _parse_cases(args)
    ctx = init_distributed(args.device, "auto")
    configure_logging(ctx.is_main_process)
    try:
        rows: list[dict[str, Any]] = []
        top_experts_by_case: dict[tuple[str, int], list[int]] = {}
        for case in cases:
            baseline = _evaluate_probe(case, ctx, args.episodes, "baseline")
            if ctx.is_main_process:
                rows.append(baseline)
                top_experts = _top_experts(
                    baseline,
                    expert_count=int(baseline.get("expert_count", 0)),
                    top_k=int(baseline.get("top_k", 1)),
                )
                top_experts_by_case[(case["lane"], case["seed"])] = top_experts
            else:
                top_experts = [0, 1]
            for expert_index in range(int(baseline.get("expert_count", 0)) or 0):
                row = _evaluate_probe(case, ctx, args.episodes, "expert_ablation", detail=str(expert_index))
                if ctx.is_main_process:
                    rows.append(row)
            fixed = _evaluate_probe(case, ctx, args.episodes, "router_override", detail="most_used_pair", fixed_experts=top_experts)
            randomized = _evaluate_probe(case, ctx, args.episodes, "route_randomization", detail="uniform_topk_random")
            if ctx.is_main_process:
                rows.append(fixed)
                rows.append(randomized)
        if not ctx.is_main_process:
            return
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_build_report(rows, top_experts_by_case, args.episodes), encoding="utf-8")
        if args.csv is not None:
            csv_path = Path(args.csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = sorted({key for row in rows for key in row})
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
