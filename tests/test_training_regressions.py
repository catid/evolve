from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import json
from torch import nn

from psmn_rl.config import load_config
from psmn_rl.rl.distributed.ddp import DistributedContext
from psmn_rl.rl.ppo.algorithm import _apply_truncation_bootstrap, current_entropy_coefficient
from psmn_rl.train import run_training


class _DummyValueModel(nn.Module):
    def forward(self, obs, state=None, done=None):
        pixels = obs["pixels"].float()
        value = pixels.flatten(1).sum(dim=1)
        return SimpleNamespace(value=value)


def test_truncation_bootstrap_uses_final_observation_value() -> None:
    model = _DummyValueModel()
    reward_t = torch.zeros(2, dtype=torch.float32)
    ctx = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        is_distributed=False,
        is_main_process=True,
        autocast_dtype=None,
    )
    final_obs = np.stack(
        [
            np.ones((2, 2, 1), dtype=np.float32),
            np.zeros((2, 2, 1), dtype=np.float32),
        ],
        axis=0,
    )
    _apply_truncation_bootstrap(
        model=model,
        next_state={},
        truncated=np.array([True, False]),
        info={"final_obs": final_obs, "_final_obs": np.array([True, False])},
        reward_t=reward_t,
        ctx=ctx,
        gamma=0.9,
    )
    assert reward_t[0].item() == pytest.approx(3.6)
    assert reward_t[1].item() == 0.0


def test_max_updates_override_is_written_to_resolved_config(tmp_path: Path) -> None:
    config = load_config("configs/baseline/minigrid_dense.yaml")
    config.system.device = "cpu"
    config.env.num_envs = 1
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 5
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 1
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "dense_override")

    run_training(config, max_updates=1)

    resolved = load_config(Path(config.logging.output_dir) / "resolved_config.yaml")
    assert resolved.ppo.total_updates == 1


def test_run_meta_records_git_provenance(tmp_path: Path) -> None:
    config = load_config("configs/baseline/minigrid_dense.yaml")
    config.system.device = "cpu"
    config.env.num_envs = 1
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 1
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "dense_meta")

    run_training(config, max_updates=1)

    run_meta = json.loads(Path(config.logging.output_dir, "run_meta.json").read_text())
    assert isinstance(run_meta["git_commit"], str)
    assert isinstance(run_meta["git_dirty"], bool)


def test_training_logs_rollout_action_diagnostics(tmp_path: Path) -> None:
    config = load_config("configs/baseline/minigrid_dense.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 1
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 1
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "dense_metrics")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "rollout/action_entropy" in last_scalar
    assert "rollout/action_max_prob" in last_scalar
    assert "rollout/action_logit_margin" in last_scalar
    assert "rollout/action_greedy_match" in last_scalar


def test_linear_entropy_schedule_with_late_start() -> None:
    config = load_config("configs/experiments/minigrid_doorkey_sare_ent1e3.yaml")
    config.ppo.ent_coef = 0.01
    config.ppo.ent_coef_final = 0.001
    config.ppo.ent_schedule = "late_linear"
    config.ppo.ent_schedule_start_fraction = 0.5

    values = [current_entropy_coefficient(config, update, total_updates=5) for update in range(1, 6)]

    assert values[0] == pytest.approx(0.01)
    assert values[1] == pytest.approx(0.01)
    assert values[-1] == pytest.approx(0.001)
    assert values[-2] > values[-1]


def test_sequence_minibatches_support_recurrent_token_gru(tmp_path: Path) -> None:
    config = load_config("configs/minigrid/main/memory_token_gru.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.ppo.sequence_minibatches = True
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_token_gru_seq")

    metrics = run_training(config, max_updates=1)

    assert metrics["global_step"] == 8.0


def test_margin_residual_training_logs_decode_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_token_gru_long_margin_residual.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_token_gru_margin_residual")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/margin_residual_gate_mean" in last_scalar
    assert "policy/margin_residual_low_margin_mean" in last_scalar


def test_por_option_action_adapter_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_por_switchy_option_adapter_small.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_por_option_adapter")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/option_action_adapter_stability" in last_scalar
    assert "policy/option_action_adapter_bias_norm" in last_scalar


def test_por_option_hidden_residual_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_por_switchy_hidden_residual_small.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_por_hidden_residual")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/option_hidden_residual_stability" in last_scalar
    assert "policy/option_hidden_residual_norm" in last_scalar


def test_por_option_action_experts_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_por_switchy_action_experts_small.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_por_action_experts")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/option_action_experts_stability" in last_scalar
    assert "policy/option_action_experts_bias_norm" in last_scalar


def test_por_option_film_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_por_switchy_option_film_small.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_por_option_film")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/option_film_stability" in last_scalar
    assert "policy/option_film_scale_norm" in last_scalar


def test_logit_gain_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_token_gru_long_logit_gain.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.ppo.sequence_minibatches = True
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_logit_gain")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/logit_gain_gate_mean" in last_scalar
    assert "policy/logit_gain_raw_norm" in last_scalar


def test_top2_rerank_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_token_gru_long_top2_rerank.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.ppo.sequence_minibatches = True
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_top2_rerank")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/top2_rerank_gate_mean" in last_scalar
    assert "policy/top2_rerank_raw_delta_mean_abs" in last_scalar


def test_por_option_context_gain_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_por_switchy_context_gain.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_context_gain")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/option_context_gain_stability" in last_scalar
    assert "policy/option_context_gain_mean" in last_scalar


def test_por_option_context_film_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_por_switchy_context_film_small.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_por_context_film")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/option_context_film_stability" in last_scalar
    assert "policy/option_context_film_scale_norm" in last_scalar


def test_por_option_context_logits_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_por_switchy_context_logits_small.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_por_context_logits")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/option_context_logits_stability" in last_scalar
    assert "policy/option_context_logits_bias_norm" in last_scalar


def test_por_option_margin_adapter_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_por_switchy_option_margin_adapter_small.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_por_option_margin_adapter")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/option_margin_adapter_stability" in last_scalar
    assert "policy/option_margin_adapter_gate_mean" in last_scalar
    assert "policy/option_margin_adapter_effective_norm" in last_scalar


def test_por_option_top2_rerank_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_por_switchy_option_top2_rerank_small.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_por_option_top2_rerank")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/option_top2_rerank_stability" in last_scalar
    assert "policy/option_top2_rerank_gate_mean" in last_scalar
    assert "policy/option_top2_rerank_effective_delta_mean_abs" in last_scalar


def test_por_option_top2_duration_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_por_switchy_option_top2_duration_small.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_por_option_top2_duration")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/option_top2_rerank_gate_signal_mean" in last_scalar
    assert "policy/option_top2_rerank_gate_mean" in last_scalar
    assert "policy/option_top2_rerank_effective_delta_mean_abs" in last_scalar


def test_por_actor_hidden_film_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_por_switchy_actor_hidden_film_small.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_por_actor_hidden_film")
    config.model.policy_option_hidden_blend_gate = True

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/option_actor_features_duration_gate" in last_scalar
    assert "policy/option_hidden_film_gate_signal_mean" in last_scalar
    assert "policy/option_hidden_film_duration_mix" in last_scalar
    assert "policy/option_hidden_branch_gates" in last_scalar
    assert "policy/option_hidden_scale_duration_mix" in last_scalar
    assert "policy/option_hidden_shift_duration_mix" in last_scalar
    assert "policy/option_hidden_low_rank" in last_scalar
    assert "policy/option_hidden_low_rank_dim" in last_scalar
    assert "policy/option_hidden_split_heads" in last_scalar
    assert "policy/option_hidden_film_scale_only" in last_scalar
    assert "policy/option_hidden_film_scale_weight" in last_scalar
    assert "policy/option_hidden_adaptive_scale_floor" in last_scalar
    assert "policy/option_hidden_scale_floor" in last_scalar
    assert "policy/option_hidden_adaptive_scale_mean" in last_scalar
    assert "policy/option_hidden_shift_compensation" in last_scalar
    assert "policy/option_hidden_shift_compensation_scale" in last_scalar
    assert "policy/option_hidden_shift_compensation_mean" in last_scalar
    assert "policy/option_hidden_blend_gate" in last_scalar
    assert "policy/option_hidden_blend_scale" in last_scalar
    assert "policy/option_hidden_blend_gate_mean" in last_scalar
    assert "policy/option_hidden_film_shift_weight" in last_scalar
    assert "policy/option_hidden_post_norm" in last_scalar
    assert "policy/option_hidden_film_scale_norm" in last_scalar


def test_por_actor_hidden_bounded_shift_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_bounded_shift2x.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_por_actor_hidden_bounded_shift")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/option_hidden_bound_shift" in last_scalar
    assert "policy/option_hidden_shift_bound_scale" in last_scalar
    assert "policy/option_hidden_film_shift_norm" in last_scalar


def test_por_actor_hidden_split_head_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_split_heads.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_por_actor_hidden_split_heads")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/option_hidden_split_heads" in last_scalar
    assert "policy/option_hidden_film_scale_norm" in last_scalar
    assert "policy/option_hidden_film_shift_norm" in last_scalar


def test_por_actor_hidden_branch_gate_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftmix95.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_por_actor_hidden_branch_gate")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/option_hidden_branch_gates" in last_scalar
    assert "policy/option_hidden_scale_duration_mix" in last_scalar
    assert "policy/option_hidden_shift_duration_mix" in last_scalar
    assert "policy/option_hidden_scale_gate_signal_mean" in last_scalar
    assert "policy/option_hidden_shift_gate_signal_mean" in last_scalar


def test_por_actor_hidden_low_rank_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_lowrank16.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_por_actor_hidden_low_rank")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/option_hidden_low_rank" in last_scalar
    assert "policy/option_hidden_low_rank_dim" in last_scalar
    assert "policy/option_hidden_film_scale_norm" in last_scalar
    assert "policy/option_hidden_film_shift_norm" in last_scalar


def test_por_actor_hidden_centered_shift_logs_metrics(tmp_path: Path) -> None:
    config = load_config("configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_centered_shift2x.yaml")
    config.system.device = "cpu"
    config.logging.tensorboard = False
    config.env.num_envs = 2
    config.env.num_eval_envs = 1
    config.ppo.rollout_steps = 4
    config.ppo.total_updates = 1
    config.ppo.update_epochs = 1
    config.ppo.minibatches = 2
    config.evaluation.episodes = 1
    config.logging.output_dir = str(tmp_path / "memory_por_actor_hidden_centered_shift")

    run_training(config, max_updates=1)

    last_scalar = {}
    for line in Path(config.logging.output_dir, "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last_scalar = row
    assert "policy/option_hidden_center_shift" in last_scalar
    assert "policy/option_hidden_center_shift_scale" in last_scalar
    assert "policy/option_hidden_center_shift_mean_abs" in last_scalar
    assert "policy/option_hidden_film_shift_norm" in last_scalar
