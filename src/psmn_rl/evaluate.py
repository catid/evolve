from __future__ import annotations

import argparse

import torch

from psmn_rl.config import load_config
from psmn_rl.envs.registry import make_vector_env
from psmn_rl.logging import configure_logging
from psmn_rl.models.factory import build_model
from psmn_rl.rl.distributed.ddp import cleanup_distributed, init_distributed
from psmn_rl.rl.ppo.algorithm import evaluate_policy
from psmn_rl.utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained PSMN RL checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--greedy", choices=("true", "false"), default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    config.system.device = args.device
    if args.episodes is not None:
        config.evaluation.episodes = args.episodes
    if args.greedy is not None:
        config.evaluation.greedy = args.greedy == "true"
    ctx = init_distributed(config.system.device, config.system.precision)
    configure_logging(ctx.is_main_process)
    set_seed(config.seed + ctx.rank, deterministic=config.system.deterministic)

    envs = make_vector_env(config.env, seed=config.seed, world_rank=ctx.rank)
    model = build_model(config.model, envs.single_observation_space, envs.single_action_space).to(ctx.device)
    envs.close()
    checkpoint = torch.load(args.checkpoint, map_location=ctx.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    try:
        metrics = evaluate_policy(
            config,
            model,
            ctx,
            episodes=args.episodes,
            greedy=config.evaluation.greedy,
            temperature=args.temperature,
        )
    finally:
        cleanup_distributed(ctx)
    if ctx.is_main_process:
        print(metrics)


if __name__ == "__main__":
    main()
