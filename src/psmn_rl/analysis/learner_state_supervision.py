from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from psmn_rl.analysis.policy_distillation import (
    DistillationBatch,
    EvalConfig,
    StudentConfig,
    TeacherConfig,
    _best_sampled_mode,
    _capture_raw_obs,
    _ce_loss,
    _discover_run_dirs,
    _format_float,
    _load_model,
    _set_trainable_parameters,
    _stack_obs,
    evaluate_modes,
)
from psmn_rl.config import dump_config, load_config
from psmn_rl.envs.registry import make_eval_env, make_reset_seeds
from psmn_rl.rl.distributed.ddp import detect_device
from psmn_rl.rl.ppo.algorithm import _episode_successes, prepare_done, prepare_obs
from psmn_rl.utils.io import get_git_commit, get_git_dirty
from psmn_rl.utils.seed import set_seed


@dataclass(slots=True)
class LoopConfig:
    rounds: int = 4
    episodes_per_round: int = 64
    max_episodes_per_round: int = 96


@dataclass(slots=True)
class LearnerStateSpec:
    name: str
    output_dir: str
    teacher: TeacherConfig
    student: StudentConfig
    loop: LoopConfig
    evaluation: EvalConfig


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _load_spec(path: str | Path) -> LearnerStateSpec:
    raw = _load_yaml(path)
    return LearnerStateSpec(
        name=str(raw["name"]),
        output_dir=str(raw["output_dir"]),
        teacher=TeacherConfig(**raw["teacher"]),
        student=StudentConfig(**raw["student"]),
        loop=LoopConfig(**raw["loop"]),
        evaluation=EvalConfig(**raw.get("evaluation", {})),
    )


def _concat_batches(left: DistillationBatch | None, right: DistillationBatch) -> DistillationBatch:
    if left is None:
        return right
    obs = {
        key: torch.cat([left.obs[key], right.obs[key]], dim=0)
        for key in left.obs
    }
    return DistillationBatch(
        obs=obs,
        actions=torch.cat([left.actions, right.actions], dim=0),
        teacher_logits=torch.cat([left.teacher_logits, right.teacher_logits], dim=0),
        weights=torch.cat([left.weights, right.weights], dim=0),
        accepted_episodes=left.accepted_episodes + right.accepted_episodes,
        episodes_seen=left.episodes_seen + right.episodes_seen,
        steps=left.steps + right.steps,
        mean_return=float(np.mean([left.mean_return, right.mean_return])),
        mean_length=float(np.mean([left.mean_length, right.mean_length])),
    )


def _collect_teacher_labels_for_student_states(
    spec: LearnerStateSpec,
    teacher: Any,
    student: Any,
    device: torch.device,
    round_index: int,
) -> DistillationBatch:
    env_config = load_config(spec.student.config)
    set_seed(env_config.seed + 30_000 + round_index, deterministic=env_config.system.deterministic)
    eval_env = make_eval_env(env_config.env, seed=env_config.seed + 300 + round_index, world_rank=0)
    reset_seed = make_reset_seeds(env_config.env.num_eval_envs, env_config.seed + 300 + round_index, world_rank=0)
    obs, _ = eval_env.reset(seed=reset_seed)
    obs_t = prepare_obs(obs, device)
    teacher_done = torch.ones(env_config.env.num_eval_envs, device=device, dtype=torch.bool)
    student_done = torch.ones(env_config.env.num_eval_envs, device=device, dtype=torch.bool)
    teacher_state = teacher.initial_state(env_config.env.num_eval_envs, device)
    student_state = student.initial_state(env_config.env.num_eval_envs, device)

    episode_obs: list[list[dict[str, np.ndarray]]] = [[] for _ in range(env_config.env.num_eval_envs)]
    episode_actions: list[list[int]] = [[] for _ in range(env_config.env.num_eval_envs)]
    episode_logits: list[list[np.ndarray]] = [[] for _ in range(env_config.env.num_eval_envs)]
    episode_returns = np.zeros(env_config.env.num_eval_envs, dtype=np.float32)
    episode_lengths = np.zeros(env_config.env.num_eval_envs, dtype=np.int32)
    collected_obs: list[dict[str, np.ndarray]] = []
    collected_actions: list[int] = []
    collected_logits: list[np.ndarray] = []
    collected_weights: list[float] = []
    collected_returns: list[float] = []
    collected_lengths: list[float] = []
    accepted_episodes = 0
    episodes_seen = 0

    try:
        while accepted_episodes < spec.loop.episodes_per_round and episodes_seen < spec.loop.max_episodes_per_round:
            with torch.inference_mode():
                teacher_output = teacher(obs_t, state=teacher_state, done=teacher_done)
                if spec.teacher.greedy:
                    teacher_action = teacher_output.logits.argmax(dim=-1)
                else:
                    teacher_dist = teacher.get_dist(teacher_output.logits, temperature=spec.teacher.temperature)
                    teacher_action = teacher_dist.sample()
                teacher_state = teacher_output.next_state

                student_output = student(obs_t, state=student_state, done=student_done)
                student_action = student_output.logits.argmax(dim=-1)
                student_state = student_output.next_state

            teacher_action_np = teacher_action.detach().cpu().numpy()
            teacher_logits_np = teacher_output.logits.detach().cpu().numpy()
            student_action_np = student_action.detach().cpu().numpy()
            raw_obs = _capture_raw_obs(obs)
            for index in range(env_config.env.num_eval_envs):
                episode_obs[index].append({key: value[index].copy() for key, value in raw_obs.items()})
                episode_actions[index].append(int(teacher_action_np[index]))
                episode_logits[index].append(teacher_logits_np[index].copy())

            next_obs, reward, terminated, truncated, info = eval_env.step(student_action_np)
            done_np = np.logical_or(terminated, truncated)
            _ = _episode_successes(reward, done_np, info)
            episode_returns += reward
            episode_lengths += 1

            for index, finished in enumerate(done_np):
                if not finished:
                    continue
                episodes_seen += 1
                accepted_episodes += 1
                episode_return = float(episode_returns[index])
                collected_returns.append(episode_return)
                collected_lengths.append(float(episode_lengths[index]))
                collected_obs.extend(episode_obs[index])
                collected_actions.extend(episode_actions[index])
                collected_logits.extend(episode_logits[index])
                collected_weights.extend([episode_return] * len(episode_actions[index]))
                episode_obs[index].clear()
                episode_actions[index].clear()
                episode_logits[index].clear()
                episode_returns[index] = 0.0
                episode_lengths[index] = 0

            obs = next_obs
            obs_t = prepare_obs(obs, device)
            teacher_done = prepare_done(done_np, device)
            student_done = prepare_done(done_np, device)
    finally:
        eval_env.close()

    return DistillationBatch(
        obs=_stack_obs(collected_obs),
        actions=torch.as_tensor(collected_actions, dtype=torch.long),
        teacher_logits=torch.as_tensor(np.stack(collected_logits, axis=0), dtype=torch.float32),
        weights=torch.as_tensor(collected_weights, dtype=torch.float32),
        accepted_episodes=accepted_episodes,
        episodes_seen=episodes_seen,
        steps=len(collected_actions),
        mean_return=float(np.mean(collected_returns)) if collected_returns else 0.0,
        mean_length=float(np.mean(collected_lengths)) if collected_lengths else 0.0,
    )


def _fine_tune_student(
    spec: LearnerStateSpec,
    student: Any,
    dataset: DistillationBatch,
    device: torch.device,
) -> dict[str, float]:
    student.train()
    trainable_params = _set_trainable_parameters(student, spec.student.target)
    optimizer = torch.optim.Adam(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=spec.student.learning_rate,
    )
    obs = {key: value.to(device) for key, value in dataset.obs.items()}
    actions = dataset.actions.to(device)
    if spec.student.weighting == "return":
        weights = dataset.weights.to(device)
    else:
        weights = torch.ones_like(dataset.weights, device=device)
    done = torch.zeros(actions.size(0), device=device, dtype=torch.bool)
    losses: list[float] = []

    for _epoch in range(spec.student.epochs):
        indices = torch.randperm(actions.size(0), device=device)
        for start in range(0, actions.size(0), spec.student.batch_size):
            batch_index = indices[start : start + spec.student.batch_size]
            obs_batch = {key: value[batch_index] for key, value in obs.items()}
            action_batch = actions[batch_index]
            done_batch = done[batch_index]
            weight_batch = weights[batch_index]
            output = student(obs_batch, state={}, done=done_batch)
            loss = _ce_loss(output.logits, action_batch, weight_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

    student.eval()
    return {
        "fine_tune/steps": float(len(losses)),
        "fine_tune/loss_final": float(losses[-1]) if losses else 0.0,
        "fine_tune/loss_mean": float(np.mean(losses)) if losses else 0.0,
        "fine_tune/trainable_params": float(trainable_params),
    }


def _write_single_run_summary(
    output_dir: Path,
    spec: LearnerStateSpec,
    teacher_variant: str,
    student_variant: str,
    before: dict[str, dict[str, float]],
    rounds: list[dict[str, Any]],
) -> None:
    final_metrics = rounds[-1]
    best_round = max(rounds, key=lambda row: float(row["after_greedy_success"]))
    before_best_mode, before_best_sampled = _best_sampled_mode(before)
    summary = {
        "name": spec.name,
        "teacher_variant": teacher_variant,
        "student_variant": student_variant,
        "teacher_config": spec.teacher.config,
        "teacher_checkpoint": spec.teacher.checkpoint,
        "student_config": spec.student.config,
        "student_checkpoint": spec.student.checkpoint,
        "target": spec.student.target,
        "weighting": spec.student.weighting,
        "before_greedy_success": float(before["greedy"].get("eval_success_rate", 0.0)),
        "before_best_sampled_mode": before_best_mode,
        "before_best_sampled_success": before_best_sampled,
        "final_greedy_success": float(final_metrics["after_greedy_success"]),
        "final_best_sampled_success": float(final_metrics["after_best_sampled_success"]),
        "best_round_index": int(best_round["round"]),
        "best_round_greedy_success": float(best_round["after_greedy_success"]),
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "rounds": rounds,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Learner-State Supervision Summary",
        "",
        f"- teacher_variant: `{teacher_variant}`",
        f"- student_variant: `{student_variant}`",
        f"- target: `{spec.student.target}`",
        f"- before_greedy_success: `{summary['before_greedy_success']:.3f}`",
        f"- best_round_greedy_success: `{summary['best_round_greedy_success']:.3f}`",
        f"- final_greedy_success: `{summary['final_greedy_success']:.3f}`",
        "",
        "| Round | Added Episodes | Aggregate Steps | After Greedy | After Best Sampled | Loss Mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rounds:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["round"])),
                    str(int(row["added_episodes"])),
                    str(int(row["aggregate_steps"])),
                    f"{row['after_greedy_success']:.3f}",
                    f"{row['after_best_sampled_success']:.3f}",
                    f"{row['fine_tune/loss_mean']:.3f}",
                ]
            )
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Learner-State Supervision Report",
        "",
        "## Summary",
        "",
        "| Run | Teacher | Student | Target | Before Greedy | Best Round Greedy | Final Greedy | Best Sampled After | Rounds |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (str(item["student_variant"]), str(item["teacher_variant"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['run_name']}`",
                    str(row["teacher_variant"]),
                    str(row["student_variant"]),
                    str(row["target"]),
                    _format_float(row["before_greedy_success"]),
                    _format_float(row["best_round_greedy_success"]),
                    _format_float(row["final_greedy_success"]),
                    _format_float(row["final_best_sampled_success"]),
                    str(len(row["rounds"])),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    for row in sorted(rows, key=lambda item: (str(item["student_variant"]), str(item["teacher_variant"]))):
        lines.append(
            f"- `{row['run_name']}` uses learner-state supervision from `{row['teacher_variant']}` into `{row['student_variant']}`, "
            f"moving greedy success from `{row['before_greedy_success']:.3f}` to best-round `{row['best_round_greedy_success']:.3f}` "
            f"and final `{row['final_greedy_success']:.3f}`."
        )
    return "\n".join(lines) + "\n"


def run_once(args: argparse.Namespace) -> None:
    spec_path = Path(args.spec)
    spec = _load_spec(spec_path)
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "spec.yaml").write_text(spec_path.read_text(encoding="utf-8"), encoding="utf-8")

    teacher_config, teacher = _load_model(spec.teacher.config, spec.teacher.checkpoint, detect_device(args.device))
    student_config, student = _load_model(spec.student.config, spec.student.checkpoint, detect_device(args.device))
    dump_config(teacher_config, output_dir / "teacher_resolved_config.yaml")
    dump_config(student_config, output_dir / "student_resolved_config.yaml")
    dump_config(student_config, output_dir / "resolved_config.yaml")
    before = evaluate_modes(spec.student.config, student, args.device, spec.evaluation.episodes)

    aggregate: DistillationBatch | None = None
    round_rows: list[dict[str, Any]] = []
    for round_index in range(1, spec.loop.rounds + 1):
        round_batch = _collect_teacher_labels_for_student_states(
            spec=spec,
            teacher=teacher,
            student=student,
            device=detect_device(args.device),
            round_index=round_index,
        )
        aggregate = _concat_batches(aggregate, round_batch)
        fine_tune_metrics = _fine_tune_student(
            spec=spec,
            student=student,
            dataset=aggregate,
            device=detect_device(args.device),
        )
        after = evaluate_modes(spec.student.config, student, args.device, spec.evaluation.episodes)
        after_best_mode, after_best_sampled = _best_sampled_mode(after)
        round_row = {
            "round": float(round_index),
            "added_episodes": float(round_batch.accepted_episodes),
            "added_steps": float(round_batch.steps),
            "aggregate_steps": float(aggregate.steps),
            "after_greedy_success": float(after["greedy"].get("eval_success_rate", 0.0)),
            "after_greedy_return": float(after["greedy"].get("eval_return", 0.0)),
            "after_best_sampled_mode": after_best_mode,
            "after_best_sampled_success": after_best_sampled,
            **fine_tune_metrics,
        }
        round_rows.append(round_row)
        torch.save(
            {
                "obs": aggregate.obs,
                "actions": aggregate.actions,
                "teacher_logits": aggregate.teacher_logits,
                "weights": aggregate.weights,
                "accepted_episodes": aggregate.accepted_episodes,
                "episodes_seen": aggregate.episodes_seen,
                "steps": aggregate.steps,
                "mean_return": aggregate.mean_return,
                "mean_length": aggregate.mean_length,
            },
            output_dir / f"round_{round_index:02d}_dataset.pt",
        )
        checkpoint = torch.load(spec.student.checkpoint, map_location="cpu", weights_only=False)
        checkpoint["model"] = student.state_dict()
        checkpoint["learner_state_supervision"] = {
            "round": round_index,
            "target": spec.student.target,
            **fine_tune_metrics,
        }
        torch.save(checkpoint, output_dir / f"round_{round_index:02d}.pt")

    teacher.eval()
    student.eval()
    torch.save(
        {
            **torch.load(spec.student.checkpoint, map_location="cpu", weights_only=False),
            "model": student.state_dict(),
        },
        output_dir / "latest.pt",
    )
    _write_single_run_summary(
        output_dir=output_dir,
        spec=spec,
        teacher_variant=teacher_config.model.variant,
        student_variant=student_config.model.variant,
        before=before,
        rounds=round_rows,
    )


def build_report(args: argparse.Namespace) -> None:
    run_dirs = _discover_run_dirs(args.paths)
    if not run_dirs:
        raise SystemExit("no learner-state supervision runs found")
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
        fieldnames = sorted({key for row in rows for key in row if key != "rounds"})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run learner-state teacher supervision experiments.")
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
