import gymnasium as gym
import torch

from psmn_rl.config import ModelConfig
from psmn_rl.models.factory import build_model


def _obs() -> tuple[gym.Env, dict[str, torch.Tensor], torch.Tensor]:
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    obs, _ = env.reset(seed=0)
    obs_t = {
        "image": torch.as_tensor(obs["image"]).unsqueeze(0),
        "direction": torch.as_tensor(obs["direction"]).unsqueeze(0),
    }
    done = torch.ones(1, dtype=torch.bool)
    return env, obs_t, done


def test_sare_reports_route_statistics() -> None:
    env, obs, done = _obs()
    model = build_model(ModelConfig(variant="sare", expert_count=4, top_k=2), env.observation_space, env.action_space)
    output = model.forward(obs, state={}, done=done)
    assert "route_entropy" in output.metrics
    assert "path_coverage_top_10" in output.metrics
    assert any(key.startswith("expert_load_") for key in output.metrics)
    env.close()


def test_treg_h_reports_hop_and_ponder_metrics() -> None:
    env, obs, done = _obs()
    model = build_model(ModelConfig(variant="treg_h", expert_count=4), env.observation_space, env.action_space)
    output = model.forward(obs, state={}, done=done)
    assert "avg_hop_count" in output.metrics
    assert "avg_halting_probability" in output.metrics
    assert "ponder_loss" in output.aux_losses
    env.close()


def test_srw_reports_relational_usage() -> None:
    env, obs, done = _obs()
    model = build_model(ModelConfig(variant="srw", expert_count=4), env.observation_space, env.action_space)
    output = model.forward(obs, state={}, done=done)
    assert "relational_usage_rate" in output.metrics
    env.close()


def test_por_updates_option_state() -> None:
    env, obs, done = _obs()
    model = build_model(ModelConfig(variant="por", option_count=4), env.observation_space, env.action_space)
    state = model.initial_state(batch_size=1, device=torch.device("cpu"))
    output = model.forward(obs, state=state, done=done)
    assert "option_probs" in output.next_state
    assert output.next_state["option_probs"].shape == (1, 4)
    assert "option_duration" in output.metrics
    env.close()
