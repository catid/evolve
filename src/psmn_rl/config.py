from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SystemConfig:
    device: str = "auto"
    precision: str = "auto"
    deterministic: bool = True
    compile: bool = False
    log_interval: int = 1
    checkpoint_interval: int = 10
    archive_checkpoints: bool = False
    resume_from: str | None = None


@dataclass(slots=True)
class EnvConfig:
    suite: str = "minigrid"
    env_id: str = "MiniGrid-DoorKey-5x5-v0"
    num_envs: int = 8
    num_eval_envs: int = 4
    max_episode_steps: int | None = None
    capture_video: bool = False
    fully_observed: bool = False
    distribution_mode: str = "easy"
    start_level: int = 0
    num_levels: int = 200


@dataclass(slots=True)
class PPOConfig:
    rollout_steps: int = 128
    total_updates: int = 64
    update_epochs: int = 4
    minibatches: int = 4
    sequence_minibatches: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_clip_coef: float = 0.2
    ent_coef: float = 0.01
    ent_coef_final: float | None = None
    ent_schedule: str = "constant"
    ent_schedule_start_fraction: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    target_kl: float | None = None


@dataclass(slots=True)
class ModelConfig:
    variant: str = "token_dense"
    checkpoint_strict: bool = True
    hidden_size: int = 128
    token_dim: int = 128
    patch_size: int = 8
    encoder_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.0
    expert_count: int = 4
    top_k: int = 2
    expert_hidden_size: int = 256
    temperature: float = 1.0
    ponder_cost: float = 0.01
    max_hops: int = 2
    halt_bias: float = 0.0
    relational_tokens: int = 4
    option_count: int = 4
    termination_bias: float = 0.0
    memory_mix: float = 0.5
    memory_gate_bias: float = 0.0
    memory_reset_bias: float = -2.0
    route_memory_scale: float = 0.5
    policy_margin_residual: bool = False
    policy_margin_residual_scale: float = 1.0
    policy_margin_threshold: float = 0.25
    policy_margin_sharpness: float = 12.0
    policy_logit_gain: bool = False
    policy_logit_gain_scale: float = 0.5
    policy_logit_gain_threshold: float = 0.25
    policy_logit_gain_sharpness: float = 12.0
    policy_top2_rerank: bool = False
    policy_top2_rerank_scale: float = 0.5
    policy_top2_rerank_threshold: float = 0.25
    policy_top2_rerank_sharpness: float = 12.0
    policy_option_margin_adapter: bool = False
    policy_option_margin_adapter_scale: float = 0.5
    policy_option_margin_threshold: float = 0.25
    policy_option_margin_sharpness: float = 12.0
    policy_option_top2_rerank: bool = False
    policy_option_top2_rerank_scale: float = 0.5
    policy_option_top2_rerank_threshold: float = 0.25
    policy_option_top2_rerank_sharpness: float = 12.0
    policy_option_top2_use_duration_gate: bool = False
    policy_option_hidden_film: bool = False
    policy_option_hidden_film_scale: float = 0.5
    policy_option_hidden_use_duration_gate: bool = False
    policy_option_hidden_duration_mix: float = 1.0
    policy_option_hidden_scale_only: bool = False
    policy_option_hidden_scale_weight: float = 1.0
    policy_option_hidden_shift_weight: float = 1.0
    policy_option_hidden_post_norm: bool = False
    por_option_action_adapter: bool = False
    por_option_action_adapter_scale: float = 1.0
    por_option_action_adapter_min_duration: float = 2.0
    por_option_action_adapter_duration_sharpness: float = 1.0
    por_option_hidden_residual: bool = False
    por_option_hidden_residual_scale: float = 1.0
    por_option_hidden_residual_min_duration: float = 2.0
    por_option_hidden_residual_duration_sharpness: float = 1.0
    por_option_action_experts: bool = False
    por_option_action_experts_scale: float = 1.0
    por_option_action_experts_min_duration: float = 2.0
    por_option_action_experts_duration_sharpness: float = 1.0
    por_option_film: bool = False
    por_option_film_scale: float = 1.0
    por_option_film_min_duration: float = 2.0
    por_option_film_duration_sharpness: float = 1.0
    por_option_context_film: bool = False
    por_option_context_film_scale: float = 1.0
    por_option_context_film_min_duration: float = 2.0
    por_option_context_film_duration_sharpness: float = 1.0
    por_option_context_logits: bool = False
    por_option_context_logits_scale: float = 0.5
    por_option_context_logits_min_duration: float = 2.0
    por_option_context_logits_duration_sharpness: float = 1.0
    por_option_margin_adapter: bool = False
    por_option_margin_adapter_min_duration: float = 2.0
    por_option_margin_adapter_duration_sharpness: float = 1.0
    por_option_top2_rerank: bool = False
    por_option_top2_rerank_min_duration: float = 2.0
    por_option_top2_rerank_duration_sharpness: float = 1.0
    por_option_actor_features: bool = False
    por_option_actor_features_min_duration: float = 2.0
    por_option_actor_features_duration_sharpness: float = 1.0
    por_option_context_gain: bool = False
    por_option_context_gain_scale: float = 0.5
    por_option_context_gain_min_duration: float = 2.0
    por_option_context_gain_duration_sharpness: float = 1.0


@dataclass(slots=True)
class LoggingConfig:
    project: str = "psmn-rl"
    run_name: str = "default"
    output_dir: str = "outputs/default"
    tensorboard: bool = True
    save_checkpoints: bool = True


@dataclass(slots=True)
class EvaluationConfig:
    episodes: int = 8
    greedy: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    seed: int = 7
    tags: list[str] = field(default_factory=list)
    system: SystemConfig = field(default_factory=SystemConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(dataclass_type: type[Any], value: dict[str, Any]) -> Any:
    return dataclass_type(**value)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return ExperimentConfig(
        seed=raw.get("seed", ExperimentConfig.seed),
        tags=raw.get("tags", []),
        system=_merge_dataclass(SystemConfig, raw.get("system", {})),
        env=_merge_dataclass(EnvConfig, raw.get("env", {})),
        ppo=_merge_dataclass(PPOConfig, raw.get("ppo", {})),
        model=_merge_dataclass(ModelConfig, raw.get("model", {})),
        logging=_merge_dataclass(LoggingConfig, raw.get("logging", {})),
        evaluation=_merge_dataclass(EvaluationConfig, raw.get("evaluation", {})),
    )


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
