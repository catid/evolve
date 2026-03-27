#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

ROOT="outputs/experiments/memory_actor_hidden_shift221_scale_probe"
REPORT_MD="outputs/reports/memory_actor_hidden_shift221_scale_probe.md"
REPORT_JSON="outputs/reports/memory_actor_hidden_shift221_scale_probe.json"
TMP_CFG_DIR="$ROOT/generated_configs"

rm -rf "$ROOT"
mkdir -p "$ROOT" "$TMP_CFG_DIR" "$(dirname "$REPORT_MD")"

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" -lt 1 ]]; then
  GPU_COUNT=1
fi

declare -a CONFIGS=(
  "configs/experiments/minigrid_memory_por_switchy_rerun.yaml"
  "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22.yaml"
  "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift221.yaml"
  "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift221_scale375.yaml"
  "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift221_scale40.yaml"
)

declare -a AVAILABLE_GPUS=()
declare -A PID_TO_GPU=()
for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
  AVAILABLE_GPUS+=("$gpu")
done

launch_job() {
  local cfg="$1"
  local gpu="${AVAILABLE_GPUS[0]}"
  AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")
  local generated_cfg="$TMP_CFG_DIR/$(basename "$cfg")"
  ./.venv/bin/python - "$cfg" "$generated_cfg" "$ROOT" <<'PY'
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
root = Path(sys.argv[3])
payload = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
run_name = str(payload["logging"]["run_name"])
payload["logging"]["output_dir"] = str(root / run_name)
dst.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY
  local run_dir
  run_dir=$(.venv/bin/python - "$generated_cfg" <<'PY'
from pathlib import Path
from psmn_rl.config import load_config
import sys

cfg = load_config(Path(sys.argv[1]))
print(cfg.logging.output_dir)
PY
)
  ./.venv/bin/python - "$generated_cfg" "$run_dir" "$gpu" <<'PY'
import hashlib
import json
import sys
import time
from pathlib import Path

cfg_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
gpu = int(sys.argv[3])
run_dir.mkdir(parents=True, exist_ok=True)
manifest = {
    "config_path": str(cfg_path),
    "config_hash": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
    "family": "memory_actor_hidden_shift221_scale_probe",
    "task": "Memory",
    "gpu": gpu,
    "output_dir": str(run_dir),
    "started_at": time.time(),
}
(run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
PY
  CUDA_VISIBLE_DEVICES="$gpu" torchrun --standalone --nproc_per_node=1 -m psmn_rl.launch --config "$generated_cfg" &
  PID_TO_GPU[$!]="$gpu"
}

reclaim_one() {
  local finished_pid
  wait -n -p finished_pid
  local status=$?
  local gpu="${PID_TO_GPU[$finished_pid]}"
  unset 'PID_TO_GPU[$finished_pid]'
  AVAILABLE_GPUS+=("$gpu")
  if (( status != 0 )); then
    exit "$status"
  fi
}

for cfg in "${CONFIGS[@]}"; do
  while [[ ${#AVAILABLE_GPUS[@]} -eq 0 ]]; do
    reclaim_one
  done
  launch_job "$cfg"
done

while [[ ${#PID_TO_GPU[@]} -gt 0 ]]; do
  reclaim_one
done

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
        "run_dir": Path("outputs/experiments/memory_actor_hidden_shift221_scale_probe/memory_por_switchy_rerun"),
        "variant": "none",
    },
    {
        "label": "por_actor_hidden_partial_shift22",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_shift221_scale_probe/memory_por_switchy_actor_hidden_partial_shift22"),
        "variant": "shift22_scale35",
    },
    {
        "label": "por_actor_hidden_partial_shift221",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_shift221_scale_probe/memory_por_switchy_actor_hidden_partial_shift221"),
        "variant": "shift221_scale35",
    },
    {
        "label": "por_actor_hidden_partial_shift221_scale375",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_shift221_scale_probe/memory_por_switchy_actor_hidden_partial_shift221_scale375"),
        "variant": "shift221_scale375",
    },
    {
        "label": "por_actor_hidden_partial_shift221_scale40",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_shift221_scale_probe/memory_por_switchy_actor_hidden_partial_shift221_scale40"),
        "variant": "shift221_scale40",
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
            "variant": run["variant"],
            "run_dir": str(run_dir),
            "film_scale": float(config.model.policy_option_hidden_film_scale),
            "shift_weight": float(config.model.policy_option_hidden_shift_weight),
            "train/episode_return": float(train_metrics.get("train/episode_return", 0.0)),
            "train/success_rate": float(train_metrics.get("train/success_rate", 0.0)),
            "eval_success_rate": float(train_metrics.get("eval_success_rate", 0.0)),
            "eval_return": float(train_metrics.get("eval_return", 0.0)),
            "throughput_fps": float(train_metrics.get("throughput_fps", 0.0)),
            "policy/option_hidden_film_gate_signal_mean": float(train_metrics.get("policy/option_hidden_film_gate_signal_mean", 0.0)),
            "policy/option_hidden_film_gate_mean": float(train_metrics.get("policy/option_hidden_film_gate_mean", 0.0)),
            "policy/option_hidden_film_scale_norm": float(train_metrics.get("policy/option_hidden_film_scale_norm", 0.0)),
            "policy/option_hidden_film_shift_norm": float(train_metrics.get("policy/option_hidden_film_shift_norm", 0.0)),
        }
        for mode_name, greedy, temperature in MODES:
            diagnostics = collect_policy_diagnostics(
                config=config,
                model=model,
                ctx=ctx,
                episodes=64,
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
por_shift221 = row("por_actor_hidden_partial_shift221")
por_scale375 = row("por_actor_hidden_partial_shift221_scale375")
por_scale40 = row("por_actor_hidden_partial_shift221_scale40")

lines = [
    "# Memory Actor-Hidden Shift221 Scale Probe",
    "",
    "- hypothesis: the sampled-strong but greedy-dead `partial_shift221` branch may be under-scaled; small scale increases at the same shift could recover greedy without giving back the lower-band lift",
    "- task: `MiniGrid-MemoryS9-v0`",
    "- fresh matched runs: `5`",
    "- same-seed architecture isolate: `seed=7` for all runs",
    "- evaluation episodes per mode: `64`",
    "",
    "## Aggregate",
    "",
    "| Label | FiLM Scale | Shift | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Train Return | Train Success | Gate Mean | Scale Norm | Shift Norm |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for item in (por_base, por_shift22, por_shift221, por_scale375, por_scale40):
    lines.append(
        f"| `{item['label']}` | `{item['film_scale']:.3f}` | `{item['shift_weight']:.4f}` | `{item['greedy/success']:.4f}` | `{item['lower_t005/success']:.4f}` | "
        f"`{item['gap_t0055/success']:.4f}` | `{item['shoulder_t008/success']:.4f}` | `{item['train/episode_return']:.4f}` | `{item['train/success_rate']:.4f}` | "
        f"`{item['policy/option_hidden_film_gate_mean']:.4f}` | `{item['policy/option_hidden_film_scale_norm']:.4f}` | `{item['policy/option_hidden_film_shift_norm']:.4f}` |"
    )

lines.extend(
    [
        "",
        "## Deltas vs References",
        "",
        "| Variant | Delta Greedy vs Shift22 | Delta Lower vs Shift22 | Delta Gap vs Shift22 | Delta Shoulder vs Shift22 | Delta Greedy vs Shift221 | Delta Lower vs Shift221 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| `por_actor_hidden_partial_shift221_scale375` | `{por_scale375['greedy/success'] - por_shift22['greedy/success']:+.4f}` | `{por_scale375['lower_t005/success'] - por_shift22['lower_t005/success']:+.4f}` | `{por_scale375['gap_t0055/success'] - por_shift22['gap_t0055/success']:+.4f}` | `{por_scale375['shoulder_t008/success'] - por_shift22['shoulder_t008/success']:+.4f}` | `{por_scale375['greedy/success'] - por_shift221['greedy/success']:+.4f}` | `{por_scale375['lower_t005/success'] - por_shift221['lower_t005/success']:+.4f}` |",
        f"| `por_actor_hidden_partial_shift221_scale40` | `{por_scale40['greedy/success'] - por_shift22['greedy/success']:+.4f}` | `{por_scale40['lower_t005/success'] - por_shift22['lower_t005/success']:+.4f}` | `{por_scale40['gap_t0055/success'] - por_shift22['gap_t0055/success']:+.4f}` | `{por_scale40['shoulder_t008/success'] - por_shift22['shoulder_t008/success']:+.4f}` | `{por_scale40['greedy/success'] - por_shift221['greedy/success']:+.4f}` | `{por_scale40['lower_t005/success'] - por_shift221['lower_t005/success']:+.4f}` |",
        "",
    ]
)

interesting = any(
    item["greedy/success"] > 0.0
    and (
        item["lower_t005/success"] > por_shift22["lower_t005/success"]
        or item["gap_t0055/success"] > por_shift22["gap_t0055/success"]
    )
    for item in (por_scale375, por_scale40)
)

if interesting:
    lines.extend(
        [
            "## Outcome",
            "",
            "At least one `partial_shift221` scale-up variant preserved nonzero greedy conversion while improving the lower or gap band over `partial_shift22`, so this surface should get a longer confirmation pass before being accepted or rejected.",
        ]
    )
else:
    lines.extend(
        [
            "## Outcome",
            "",
            "Raising FiLM scale on the `partial_shift221` branch did not produce a better point than `partial_shift22`. Any surviving sampled behavior failed to recover greedy while preserving the lower-band advantage.",
        ]
    )

REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
