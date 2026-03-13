from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from psmn_rl.config import ExperimentConfig, load_config
from psmn_rl.envs.registry import make_vector_env
from psmn_rl.logging import configure_logging
from psmn_rl.models.factory import build_model
from psmn_rl.rl.distributed.ddp import cleanup_distributed, init_distributed, maybe_wrap_ddp, unwrap_ddp
from psmn_rl.rl.ppo.algorithm import train
from psmn_rl.utils.seed import restore_rng_state, set_seed


@dataclass(slots=True)
class ResumeState:
    update: int = 0
    step: int = 0


def maybe_resume_training(
    config: ExperimentConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> ResumeState:
    if not config.system.resume_from:
        return ResumeState()
    checkpoint = torch.load(config.system.resume_from, map_location=device, weights_only=False)
    unwrap_ddp(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng_state(checkpoint.get("rng_state"))
    return ResumeState(update=int(checkpoint.get("update", 0)), step=int(checkpoint.get("step", 0)))


def resolve_output_dir(config: ExperimentConfig, config_path: str, output_dir: str | None) -> None:
    if output_dir is not None:
        config.logging.output_dir = output_dir
        return
    if config.logging.output_dir == "outputs/default":
        config.logging.output_dir = str(Path("outputs") / Path(config_path).stem)


def run_training(config: ExperimentConfig, max_updates: int | None = None) -> dict[str, float]:
    if max_updates is not None:
        config.ppo.total_updates = max_updates
    ctx = init_distributed(config.system.device, config.system.precision)
    configure_logging(ctx.is_main_process)
    set_seed(config.seed + ctx.rank, deterministic=config.system.deterministic)

    envs = make_vector_env(config.env, seed=config.seed, world_rank=ctx.rank)
    model = build_model(config.model, envs.single_observation_space, envs.single_action_space)
    envs.close()
    model.to(ctx.device)
    if config.system.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    model = maybe_wrap_ddp(model, ctx)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.learning_rate, eps=1e-5)
    resume_state = maybe_resume_training(config, model, optimizer, ctx.device)

    try:
        result = train(
            config,
            model,
            optimizer,
            ctx,
            max_updates=max_updates,
            start_update=resume_state.update,
            start_step=resume_state.step,
        )
    finally:
        cleanup_distributed(ctx)
    return result.final_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO agents for PSMN RL experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--max-updates", type=int, default=None, help="Override the number of PPO updates")
    parser.add_argument("--device", type=str, default=None, help="Override device selection")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from a saved checkpoint path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config.system.device = args.device
    if args.resume_from is not None:
        config.system.resume_from = args.resume_from
    resolve_output_dir(config, args.config, args.output_dir)
    run_training(config, max_updates=args.max_updates)


if __name__ == "__main__":
    main()
