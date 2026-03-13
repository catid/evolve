import pytest

from psmn_rl.config import EnvConfig
from psmn_rl.envs.registry import make_vector_env
from psmn_rl.envs import procgen as procgen_module


def test_procgen_missing_package_raises_clear_error(monkeypatch) -> None:
    config = EnvConfig(
        suite="procgen",
        env_id="procgen_gym/procgen-coinrun-v0",
        num_envs=1,
        num_eval_envs=1,
    )
    monkeypatch.setattr(
        procgen_module,
        "ensure_procgen_available",
        lambda: (_ for _ in ()).throw(
            RuntimeError("Procgen support requires a Gymnasium-compatible Procgen package")
        ),
    )
    with pytest.raises(RuntimeError, match="Procgen support requires a Gymnasium-compatible Procgen package"):
        env = make_vector_env(config, seed=7)
        try:
            env.reset(seed=7)
        finally:
            env.close()
