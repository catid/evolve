#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

REPORT_MD="outputs/reports/memory_actor_hidden_adaptive_scale_floor_confirmation.md"
REPORT_JSON="outputs/reports/memory_actor_hidden_adaptive_scale_floor_confirmation.json"

mkdir -p "$(dirname "$REPORT_MD")"

./.venv/bin/python - "$REPORT_MD" "$REPORT_JSON" <<'PY'
import json
from pathlib import Path
from typing import Any

import torch

from psmn_rl.config import load_config
from psmn_rl.envs.registry import make_vector_env
from psmn_rl.logging import configure_logging
from psmn_rl.models.factory import build_model
from psmn_rl.rl.distributed.ddp import cleanup_distributed, init_distributed
from psmn_rl.rl.ppo.algorithm import collect_policy_diagnostics
from psmn_rl.utils.seed import set_seed

REPORT_MD = Path(__import__("sys").argv[1])
REPORT_JSON = Path(__import__("sys").argv[2])

RUNS = [
    {
        "label": "por_base",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_adaptive_scale_floor_probe/memory_por_switchy_rerun"),
    },
    {
        "label": "por_actor_hidden_partial_shift22",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_adaptive_scale_floor_probe/memory_por_switchy_actor_hidden_partial_shift22"),
    },
    {
        "label": "por_actor_hidden_partial_shift22_adaptive_floor25",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_adaptive_scale_floor_probe/memory_por_switchy_actor_hidden_partial_shift22_adaptive_floor25"),
    },
]

MODES = [
    ("greedy", True, 1.0),
    ("lower_t005", False, 0.05),
    ("gap_t0055", False, 0.055),
    ("shoulder_t008", False, 0.08),
]


def load_last_scalar(run_dir: Path) -> dict[str, float]:
    metrics_path = run_dir / "metrics.jsonl"
    last: dict[str, Any] = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last = row
    return {key: float(value) for key, value in last.items() if isinstance(value, (int, float))}


ctx = init_distributed("auto", "auto")
configure_logging(ctx.is_main_process)
rows: list[dict[str, Any]] = []
try:
    for run in RUNS:
        run_dir = run["run_dir"]
        config = load_config(run_dir / "resolved_config.yaml")
        config.system.device = "auto"
        config.logging.tensorboard = False
        set_seed(config.seed + ctx.rank, deterministic=config.system.deterministic)
        envs = make_vector_env(config.env, seed=config.seed, world_rank=ctx.rank)
        model = build_model(config.model, envs.single_observation_space, envs.single_action_space).to(ctx.device)
        envs.close()
        checkpoint = torch.load(run_dir / "latest.pt", map_location=ctx.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        train_metrics = load_last_scalar(run_dir)
        row: dict[str, Any] = {
            "label": run["label"],
            "train/episode_return": float(train_metrics.get("train/episode_return", 0.0)),
            "train/success_rate": float(train_metrics.get("train/success_rate", 0.0)),
            "policy/option_hidden_film_gate_signal_mean": float(train_metrics.get("policy/option_hidden_film_gate_signal_mean", 0.0)),
            "policy/option_hidden_adaptive_scale_mean": float(train_metrics.get("policy/option_hidden_adaptive_scale_mean", 0.0)),
            "policy/option_hidden_film_scale_norm": float(train_metrics.get("policy/option_hidden_film_scale_norm", 0.0)),
            "policy/option_hidden_film_shift_norm": float(train_metrics.get("policy/option_hidden_film_shift_norm", 0.0)),
        }
        for mode_name, greedy, temperature in MODES:
            diagnostics = collect_policy_diagnostics(
                config=config,
                model=model,
                ctx=ctx,
                episodes=256,
                greedy=greedy,
                temperature=temperature,
                trace_limit=0,
            )
            row[f"{mode_name}/success"] = float(diagnostics.metrics.get("eval_success_rate", 0.0))
            row[f"{mode_name}/return"] = float(diagnostics.metrics.get("eval_return", 0.0))
            row[f"{mode_name}/margin"] = float(diagnostics.metrics.get("eval/action_logit_margin", 0.0))
        rows.append(row)
finally:
    cleanup_distributed(ctx)

REPORT_JSON.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")


def row(label: str) -> dict[str, Any]:
    return next(item for item in rows if item["label"] == label)


por_base = row("por_base")
por_shift22 = row("por_actor_hidden_partial_shift22")
por_floor25 = row("por_actor_hidden_partial_shift22_adaptive_floor25")

lines = [
    "# Memory Actor-Hidden Adaptive Scale-Floor Confirmation",
    "",
    "- follow-up to the 64-episode screen in `memory_actor_hidden_adaptive_scale_floor_probe.md`",
    "- task: `MiniGrid-MemoryS9-v0`",
    "- evaluation episodes per mode: `256`",
    "- matched rerun roots: base POR, incumbent `partial_shift22`, adaptive-floor survivor `adaptive_floor25`",
    "",
    "## Aggregate",
    "",
    "| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Adaptive Scale Mean | Scale Norm | Shift Norm |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for item in (por_base, por_shift22, por_floor25):
    lines.append(
        f"| `{item['label']}` | `{item['greedy/success']:.4f}` | `{item['lower_t005/success']:.4f}` | `{item['gap_t0055/success']:.4f}` | `{item['shoulder_t008/success']:.4f}` | "
        f"`{item['greedy/margin']:.4f}` | `{item['gap_t0055/margin']:.4f}` | `{item['shoulder_t008/margin']:.4f}` | "
        f"`{item['train/episode_return']:.4f}` | `{item['train/success_rate']:.4f}` | `{item['policy/option_hidden_film_gate_signal_mean']:.4f}` | `{item['policy/option_hidden_adaptive_scale_mean']:.4f}` | `{item['policy/option_hidden_film_scale_norm']:.4f}` | `{item['policy/option_hidden_film_shift_norm']:.4f}` |"
    )

lines.extend(
    [
        "",
        "## Deltas vs Partial-Shift22",
        "",
        f"- `adaptive_floor25`: greedy `{por_floor25['greedy/success'] - por_shift22['greedy/success']:+.4f}`, lower `{por_floor25['lower_t005/success'] - por_shift22['lower_t005/success']:+.4f}`, gap `{por_floor25['gap_t0055/success'] - por_shift22['gap_t0055/success']:+.4f}`, shoulder `{por_floor25['shoulder_t008/success'] - por_shift22['shoulder_t008/success']:+.4f}`",
        "",
        "## Conclusion",
        "",
    ]
)

if por_floor25["greedy/success"] > 0.0 and (
    por_floor25["gap_t0055/success"] > por_shift22["gap_t0055/success"]
    or por_floor25["lower_t005/success"] > por_shift22["lower_t005/success"]
    or por_floor25["shoulder_t008/success"] > por_shift22["shoulder_t008/success"]
):
    lines.append(
        f"- `adaptive_floor25` remains alive at `greedy/lower/gap/shoulder = {por_floor25['greedy/success']:.4f} / {por_floor25['lower_t005/success']:.4f} / {por_floor25['gap_t0055/success']:.4f} / {por_floor25['shoulder_t008/success']:.4f}`"
    )
    lines.append(
        "- interpretation: duration-conditioned scale softening creates a real greedy-versus-sampled trade-off that survives beyond the 64-episode screen"
    )
else:
    lines.append(
        f"- `adaptive_floor25` does not hold up as a clear improvement over `partial_shift22` at 256 episodes: `greedy/lower/gap/shoulder = {por_floor25['greedy/success']:.4f} / {por_floor25['lower_t005/success']:.4f} / {por_floor25['gap_t0055/success']:.4f} / {por_floor25['shoulder_t008/success']:.4f}` versus incumbent `{por_shift22['greedy/success']:.4f} / {por_shift22['lower_t005/success']:.4f} / {por_shift22['gap_t0055/success']:.4f} / {por_shift22['shoulder_t008/success']:.4f}`"
    )
    lines.append(
        "- interpretation: the adaptive scale-floor effect is either too weak or too trade-off-heavy to replace the incumbent local point"
    )

REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
