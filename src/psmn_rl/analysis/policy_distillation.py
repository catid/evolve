from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from psmn_rl.analysis.policy_diagnostics import DEFAULT_MODES
from psmn_rl.config import dump_config, load_config
from psmn_rl.envs.registry import make_eval_env, make_reset_seeds, make_vector_env
from psmn_rl.models.common import ActorCriticModel
from psmn_rl.models.factory import build_model
from psmn_rl.rl.distributed.ddp import DistributedContext, detect_device
from psmn_rl.rl.ppo.algorithm import _episode_successes, collect_policy_diagnostics, prepare_done, prepare_obs
from psmn_rl.utils.io import get_git_commit, get_git_dirty
from psmn_rl.utils.seed import set_seed


@dataclass(slots=True)
class TeacherConfig:
    config: str
    checkpoint: str
    greedy: bool = True
    temperature: float = 1.0


@dataclass(slots=True)
class StudentConfig:
    config: str
    checkpoint: str
    target: str = "policy_head"
    loss: str = "ce"
    ce_weight: float = 1.0
    kl_weight: float = 1.0
    weighting: str = "uniform"
    learning_rate: float = 1e-4
    batch_size: int = 128
    epochs: int = 8


@dataclass(slots=True)
class HarvestConfig:
    success_only: bool = True
    target_episodes: int = 64
    max_episodes: int = 256


@dataclass(slots=True)
class EvalConfig:
    episodes: int = 32


@dataclass(slots=True)
class DistillationSpec:
    name: str
    output_dir: str
    teacher: TeacherConfig
    student: StudentConfig
    harvest: HarvestConfig
    evaluation: EvalConfig


@dataclass(slots=True)
class DistillationBatch:
    obs: dict[str, torch.Tensor]
    actions: torch.Tensor
    teacher_logits: torch.Tensor
    weights: torch.Tensor
    accepted_episodes: int
    episodes_seen: int
    steps: int
    mean_return: float
    mean_length: float
    teacher_confidence: torch.Tensor | None = None


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _load_spec(path: str | Path) -> DistillationSpec:
    raw = _load_yaml(path)
    return DistillationSpec(
        name=str(raw["name"]),
        output_dir=str(raw["output_dir"]),
        teacher=TeacherConfig(**raw["teacher"]),
        student=StudentConfig(**raw["student"]),
        harvest=HarvestConfig(**raw["harvest"]),
        evaluation=EvalConfig(**raw.get("evaluation", {})),
    )


def _local_ctx(device: torch.device) -> DistributedContext:
    return DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=device,
        is_distributed=False,
        is_main_process=True,
        autocast_dtype=None,
    )


def _stack_obs(samples: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    first = samples[0]
    return {
        key: torch.as_tensor(np.stack([sample[key] for sample in samples], axis=0))
        for key in first
    }


def _capture_raw_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value)
        for key, value in obs.items()
        if key in ("image", "direction", "pixels")
    }


def _set_trainable_parameters(model: ActorCriticModel, target: str) -> int:
    if target == "full_model":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return sum(parameter.numel() for parameter in model.parameters())

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.policy_head.parameters():
        parameter.requires_grad_(True)
    if target == "policy_head_plus_last_shared":
        core = model.core
        if hasattr(core, "blocks") and getattr(core, "blocks"):
            for parameter in core.blocks[-1].parameters():
                parameter.requires_grad_(True)
            norm = getattr(core, "norm", None)
            if norm is not None:
                for parameter in norm.parameters():
                    parameter.requires_grad_(True)
        elif hasattr(core, "bank"):
            for attr in ("query", "bank", "output_norm"):
                module = getattr(core, attr, None)
                if module is None:
                    continue
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            expert_keys = getattr(core, "expert_keys", None)
            if isinstance(expert_keys, torch.nn.Parameter):
                expert_keys.requires_grad_(True)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _weighted_loss(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    norm = weights.mean().clamp_min(1e-8)
    return (losses * (weights / norm)).mean()


def _ce_loss(student_logits: torch.Tensor, teacher_actions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return _weighted_loss(F.cross_entropy(student_logits, teacher_actions, reduction="none"), weights)


def _kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    per_item = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return _weighted_loss(per_item, weights)


def _teacher_confidence(teacher_logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(teacher_logits, dim=-1).max(dim=-1).values


def _dataset_weights(spec: StudentConfig, dataset: DistillationBatch, device: torch.device) -> torch.Tensor:
    weights = torch.ones_like(dataset.weights, device=device)
    if spec.weighting == "return":
        weights = dataset.weights.to(device)
    elif spec.weighting == "teacher_confidence":
        if dataset.teacher_confidence is None:
            raise ValueError("teacher_confidence weighting requested but dataset lacks teacher_confidence")
        weights = dataset.teacher_confidence.to(device)
    elif spec.weighting != "uniform":
        raise ValueError(f"unsupported weighting mode: {spec.weighting}")
    return weights


def _student_loss(
    student_cfg: StudentConfig,
    student_logits: torch.Tensor,
    teacher_actions: torch.Tensor,
    teacher_logits: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if student_cfg.loss == "ce":
        return _ce_loss(student_logits, teacher_actions, weights)
    if student_cfg.loss == "kl":
        return _kl_loss(student_logits, teacher_logits, weights)
    if student_cfg.loss == "ce_kl":
        ce = _ce_loss(student_logits, teacher_actions, weights)
        kl = _kl_loss(student_logits, teacher_logits, weights)
        return student_cfg.ce_weight * ce + student_cfg.kl_weight * kl
    raise ValueError(f"unsupported student loss: {student_cfg.loss}")


def _load_model(config_path: str, checkpoint_path: str, device: torch.device) -> tuple[Any, ActorCriticModel]:
    config = load_config(config_path)
    envs = make_vector_env(config.env, seed=config.seed)
    model = build_model(config.model, envs.single_observation_space, envs.single_action_space)
    envs.close()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return config, model


def harvest_teacher_dataset(spec: DistillationSpec, device: str) -> DistillationBatch:
    device_t = detect_device(device)
    teacher_config, teacher = _load_model(spec.teacher.config, spec.teacher.checkpoint, device_t)
    teacher.eval()
    set_seed(load_config(spec.teacher.config).seed + 20_001, deterministic=teacher_config.system.deterministic)

    eval_env = make_eval_env(teacher_config.env, seed=teacher_config.seed + 111, world_rank=0)
    reset_seed = make_reset_seeds(teacher_config.env.num_eval_envs, teacher_config.seed + 111, world_rank=0)
    obs, _ = eval_env.reset(seed=reset_seed)
    obs_t = prepare_obs(obs, device_t)
    done_t = torch.ones(teacher_config.env.num_eval_envs, device=device_t, dtype=torch.bool)
    state = teacher.initial_state(teacher_config.env.num_eval_envs, device_t)

    episode_obs: list[list[dict[str, np.ndarray]]] = [[] for _ in range(teacher_config.env.num_eval_envs)]
    episode_actions: list[list[int]] = [[] for _ in range(teacher_config.env.num_eval_envs)]
    episode_logits: list[list[np.ndarray]] = [[] for _ in range(teacher_config.env.num_eval_envs)]
    episode_returns = np.zeros(teacher_config.env.num_eval_envs, dtype=np.float32)
    episode_lengths = np.zeros(teacher_config.env.num_eval_envs, dtype=np.int32)
    accepted_obs: list[dict[str, np.ndarray]] = []
    accepted_actions: list[int] = []
    accepted_logits: list[np.ndarray] = []
    accepted_weights: list[float] = []
    accepted_returns: list[float] = []
    accepted_lengths: list[float] = []
    accepted_episodes = 0
    episodes_seen = 0

    try:
        while accepted_episodes < spec.harvest.target_episodes and episodes_seen < spec.harvest.max_episodes:
            with torch.inference_mode():
                output = teacher(obs_t, state=state, done=done_t)
                dist = teacher.get_dist(output.logits, temperature=spec.teacher.temperature)
                if spec.teacher.greedy:
                    action = output.logits.argmax(dim=-1)
                else:
                    action = dist.sample()
                state = output.next_state
            action_np = action.detach().cpu().numpy()
            logits_np = output.logits.detach().cpu().numpy()
            raw_obs = _capture_raw_obs(obs)
            for index in range(teacher_config.env.num_eval_envs):
                episode_obs[index].append({key: value[index].copy() for key, value in raw_obs.items()})
                episode_actions[index].append(int(action_np[index]))
                episode_logits[index].append(logits_np[index].copy())

            next_obs, reward, terminated, truncated, info = eval_env.step(action_np)
            done_np = np.logical_or(terminated, truncated)
            step_success = _episode_successes(reward, done_np, info)
            episode_returns += reward
            episode_lengths += 1
            for index, finished in enumerate(done_np):
                if not finished:
                    continue
                episodes_seen += 1
                keep_episode = (not spec.harvest.success_only) or (step_success[index] > 0.0)
                if keep_episode:
                    episode_return = float(episode_returns[index])
                    accepted_episodes += 1
                    accepted_returns.append(episode_return)
                    accepted_lengths.append(float(episode_lengths[index]))
                    accepted_obs.extend(episode_obs[index])
                    accepted_actions.extend(episode_actions[index])
                    accepted_logits.extend(episode_logits[index])
                    accepted_weights.extend([episode_return] * len(episode_actions[index]))
                episode_obs[index].clear()
                episode_actions[index].clear()
                episode_logits[index].clear()
                episode_returns[index] = 0.0
                episode_lengths[index] = 0
            obs = next_obs
            obs_t = prepare_obs(obs, device_t)
            done_t = prepare_done(done_np, device_t)
    finally:
        eval_env.close()

    if not accepted_returns:
        raise RuntimeError("teacher harvest collected no accepted episodes")
    return DistillationBatch(
        obs=_stack_obs(accepted_obs),
        actions=torch.as_tensor(accepted_actions, dtype=torch.long),
        teacher_logits=torch.as_tensor(np.stack(accepted_logits, axis=0), dtype=torch.float32),
        weights=torch.as_tensor(accepted_weights, dtype=torch.float32),
        teacher_confidence=_teacher_confidence(torch.as_tensor(np.stack(accepted_logits, axis=0), dtype=torch.float32)),
        accepted_episodes=accepted_episodes,
        episodes_seen=episodes_seen,
        steps=len(accepted_actions),
        mean_return=float(np.mean(accepted_returns)),
        mean_length=float(np.mean(accepted_lengths)),
    )


def fine_tune_student(
    spec: DistillationSpec,
    dataset: DistillationBatch,
    device: str,
) -> tuple[ActorCriticModel, dict[str, float]]:
    device_t = detect_device(device)
    student_config, student = _load_model(spec.student.config, spec.student.checkpoint, device_t)
    set_seed(student_config.seed + 20_002, deterministic=student_config.system.deterministic)
    student.train()
    trainable_params = _set_trainable_parameters(student, spec.student.target)
    optimizer = torch.optim.Adam(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=spec.student.learning_rate,
    )

    obs = {key: value.to(device_t) for key, value in dataset.obs.items()}
    actions = dataset.actions.to(device_t)
    teacher_logits = dataset.teacher_logits.to(device_t)
    weights = _dataset_weights(spec.student, dataset, device_t)
    done = torch.zeros(actions.size(0), device=device_t, dtype=torch.bool)
    losses: list[float] = []

    for _epoch in range(spec.student.epochs):
        indices = torch.randperm(actions.size(0), device=device_t)
        for start in range(0, actions.size(0), spec.student.batch_size):
            batch_index = indices[start : start + spec.student.batch_size]
            obs_batch = {key: value[batch_index] for key, value in obs.items()}
            action_batch = actions[batch_index]
            teacher_logits_batch = teacher_logits[batch_index]
            weight_batch = weights[batch_index]
            done_batch = done[batch_index]

            output = student(obs_batch, state={}, done=done_batch)
            loss = _student_loss(
                spec.student,
                output.logits,
                action_batch,
                teacher_logits_batch,
                weight_batch,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

    student.eval()
    return student, {
        "fine_tune/steps": float(len(losses)),
        "fine_tune/loss_final": float(losses[-1]) if losses else 0.0,
        "fine_tune/loss_mean": float(np.mean(losses)) if losses else 0.0,
        "fine_tune/trainable_params": float(trainable_params),
    }


def evaluate_modes(config_path: str, model: ActorCriticModel, device: str, episodes: int) -> dict[str, dict[str, float]]:
    config = load_config(config_path)
    config.system.device = device
    ctx = _local_ctx(detect_device(device))
    set_seed(config.seed, deterministic=config.system.deterministic)
    metrics: dict[str, dict[str, float]] = {}
    for mode_name, greedy, temperature in DEFAULT_MODES:
        diagnostics = collect_policy_diagnostics(
            config=config,
            model=model,
            ctx=ctx,
            episodes=episodes,
            greedy=greedy,
            temperature=temperature,
            trace_limit=0,
        )
        metrics[mode_name] = diagnostics.metrics
    return metrics


def _best_sampled_mode(metrics_by_mode: dict[str, dict[str, float]]) -> tuple[str, float]:
    sampled = [(mode, metrics) for mode, metrics in metrics_by_mode.items() if mode != "greedy"]
    best_mode, best_metrics = max(sampled, key=lambda item: item[1].get("eval_success_rate", 0.0))
    return best_mode, float(best_metrics.get("eval_success_rate", 0.0))


def _write_single_run_summary(
    output_dir: Path,
    spec: DistillationSpec,
    teacher_variant: str,
    student_variant: str,
    dataset: DistillationBatch,
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
    fine_tune_metrics: dict[str, float],
) -> None:
    before_best_mode, before_best_sampled = _best_sampled_mode(before)
    after_best_mode, after_best_sampled = _best_sampled_mode(after)
    summary = {
        "name": spec.name,
        "teacher_variant": teacher_variant,
        "student_variant": student_variant,
        "teacher_config": spec.teacher.config,
        "teacher_checkpoint": spec.teacher.checkpoint,
        "teacher_greedy": spec.teacher.greedy,
        "teacher_temperature": spec.teacher.temperature,
        "student_config": spec.student.config,
        "student_checkpoint": spec.student.checkpoint,
        "target": spec.student.target,
        "loss": spec.student.loss,
        "weighting": spec.student.weighting,
        "harvest_success_only": spec.harvest.success_only,
        "harvest_target_episodes": spec.harvest.target_episodes,
        "harvest_accepted_episodes": dataset.accepted_episodes,
        "harvest_episodes_seen": dataset.episodes_seen,
        "harvest_steps": dataset.steps,
        "harvest_mean_return": dataset.mean_return,
        "harvest_mean_length": dataset.mean_length,
        "before_greedy_success": float(before["greedy"].get("eval_success_rate", 0.0)),
        "before_greedy_return": float(before["greedy"].get("eval_return", 0.0)),
        "before_greedy_max_prob": float(before["greedy"].get("eval/action_max_prob", 0.0)),
        "before_greedy_margin": float(before["greedy"].get("eval/action_logit_margin", 0.0)),
        "before_best_sampled_mode": before_best_mode,
        "before_best_sampled_success": before_best_sampled,
        "after_greedy_success": float(after["greedy"].get("eval_success_rate", 0.0)),
        "after_greedy_return": float(after["greedy"].get("eval_return", 0.0)),
        "after_greedy_max_prob": float(after["greedy"].get("eval/action_max_prob", 0.0)),
        "after_greedy_margin": float(after["greedy"].get("eval/action_logit_margin", 0.0)),
        "after_best_sampled_mode": after_best_mode,
        "after_best_sampled_success": after_best_sampled,
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        **fine_tune_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Policy Distillation Summary",
                "",
                f"- teacher_variant: `{teacher_variant}`",
                f"- student_variant: `{student_variant}`",
                f"- target: `{spec.student.target}`",
                f"- loss: `{spec.student.loss}`",
                f"- weighting: `{spec.student.weighting}`",
                f"- before_greedy_success: `{summary['before_greedy_success']:.3f}`",
                f"- after_greedy_success: `{summary['after_greedy_success']:.3f}`",
                f"- before_best_sampled_mode: `{before_best_mode}`",
                f"- after_best_sampled_mode: `{after_best_mode}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _discover_run_dirs(paths: list[str]) -> list[Path]:
    discovered: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if (path / "summary.json").exists():
            discovered.append(path)
            continue
        discovered.extend(sorted(parent.parent for parent in path.glob("*/summary.json")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in discovered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _build_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Policy Distillation Report",
        "",
        "## Summary",
        "",
        "| Run | Teacher | Student | Target | Loss | Before Greedy | After Greedy | Before Best Sampled | After Best Sampled | Harvest Episodes | Trainable Params |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (str(item["student_variant"]), str(item["teacher_variant"]), str(item["target"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['run_name']}`",
                    str(row["teacher_variant"]),
                    str(row["student_variant"]),
                    str(row["target"]),
                    str(row["loss"]),
                    _format_float(row["before_greedy_success"]),
                    _format_float(row["after_greedy_success"]),
                    _format_float(row["before_best_sampled_success"]),
                    _format_float(row["after_best_sampled_success"]),
                    str(int(row["harvest_accepted_episodes"])),
                    f"{row['fine_tune/trainable_params']:.0f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Best Run By Student/Teacher Pair", "", "| Teacher | Student | Best Target | Best Loss | Best Greedy Success | Greedy Delta | Best Sampled Success |", "| --- | --- | --- | --- | ---: | ---: | ---: |"])
    best_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["teacher_variant"]), str(row["student_variant"]))
        current = best_rows.get(key)
        candidate = (
            float(row["after_greedy_success"]),
            float(row["after_best_sampled_success"]),
        )
        incumbent = (
            float(current["after_greedy_success"]),
            float(current["after_best_sampled_success"]),
        ) if current is not None else None
        if current is None or candidate > incumbent:
            best_rows[key] = row
    for (teacher_variant, student_variant), row in sorted(best_rows.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    teacher_variant,
                    student_variant,
                    str(row["target"]),
                    str(row["loss"]),
                    _format_float(row["after_greedy_success"]),
                    _format_float(float(row["after_greedy_success"]) - float(row["before_greedy_success"])),
                    _format_float(row["after_best_sampled_success"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    for row in sorted(rows, key=lambda item: (str(item["student_variant"]), str(item["teacher_variant"]), str(item["target"]))):
        lines.append(
            f"- `{row['run_name']}` uses `{row['teacher_variant']}` as teacher and `{row['student_variant']}` as student, "
            f"moving greedy success from `{row['before_greedy_success']:.3f}` to `{row['after_greedy_success']:.3f}` "
            f"and best sampled success from `{row['before_best_sampled_success']:.3f}` to `{row['after_best_sampled_success']:.3f}`."
        )
    return "\n".join(lines) + "\n"


def run_once(args: argparse.Namespace) -> None:
    spec_path = Path(args.spec)
    spec = _load_spec(spec_path)
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "spec.yaml").write_text(spec_path.read_text(encoding="utf-8"), encoding="utf-8")

    teacher_config = load_config(spec.teacher.config)
    student_config = load_config(spec.student.config)
    dump_config(teacher_config, output_dir / "teacher_resolved_config.yaml")
    dump_config(student_config, output_dir / "student_resolved_config.yaml")
    dump_config(student_config, output_dir / "resolved_config.yaml")

    before_model_env = make_vector_env(student_config.env, seed=student_config.seed)
    before_model = build_model(
        student_config.model,
        before_model_env.single_observation_space,
        before_model_env.single_action_space,
    )
    before_model_env.close()
    before_model.load_state_dict(torch.load(spec.student.checkpoint, map_location="cpu", weights_only=False)["model"])
    before_model.to(detect_device(args.device))
    before = evaluate_modes(spec.student.config, before_model, args.device, spec.evaluation.episodes)

    dataset = harvest_teacher_dataset(spec, args.device)
    torch.save(
        {
            "obs": dataset.obs,
            "actions": dataset.actions,
            "teacher_logits": dataset.teacher_logits,
            "weights": dataset.weights,
            "accepted_episodes": dataset.accepted_episodes,
            "episodes_seen": dataset.episodes_seen,
            "steps": dataset.steps,
            "mean_return": dataset.mean_return,
            "mean_length": dataset.mean_length,
        },
        output_dir / "harvest.pt",
    )

    model, fine_tune_metrics = fine_tune_student(spec, dataset, args.device)
    after = evaluate_modes(spec.student.config, model, args.device, spec.evaluation.episodes)

    checkpoint = torch.load(spec.student.checkpoint, map_location="cpu", weights_only=False)
    checkpoint["model"] = model.state_dict()
    checkpoint["policy_distillation"] = {
        "spec_name": spec.name,
        "teacher_config": spec.teacher.config,
        "teacher_checkpoint": spec.teacher.checkpoint,
        "target": spec.student.target,
        "loss": spec.student.loss,
        "weighting": spec.student.weighting,
        **fine_tune_metrics,
    }
    torch.save(checkpoint, output_dir / "latest.pt")
    _write_single_run_summary(
        output_dir=output_dir,
        spec=spec,
        teacher_variant=teacher_config.model.variant,
        student_variant=student_config.model.variant,
        dataset=dataset,
        before=before,
        after=after,
        fine_tune_metrics=fine_tune_metrics,
    )


def build_report(args: argparse.Namespace) -> None:
    run_dirs = _discover_run_dirs(args.paths)
    if not run_dirs:
        raise SystemExit("no policy distillation runs found")
    rows = []
    for run_dir in run_dirs:
        row = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        row["run_name"] = run_dir.name
        rows.append(row)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_build_report(rows), encoding="utf-8")
    if args.csv is not None:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run teacher-guided offline policy distillation experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--spec", required=True)
    run_parser.add_argument("--device", default="cpu")

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("paths", nargs="+")
    report_parser.add_argument("--output", required=True)
    report_parser.add_argument("--csv", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run_once(args)
        return
    build_report(args)


if __name__ == "__main__":
    main()
