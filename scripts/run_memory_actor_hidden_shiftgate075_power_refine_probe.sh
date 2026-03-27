#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

ROOT="outputs/experiments/memory_actor_hidden_shiftgate075_power_refine_probe"
REPORT_MD="outputs/reports/memory_actor_hidden_shiftgate075_power_refine_probe.md"
REPORT_JSON="outputs/reports/memory_actor_hidden_shiftgate075_power_refine_probe.json"
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
  "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075.yaml"
  "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate070.yaml"
  "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate085.yaml"
)

launch_config() {
  local cfg="$1"
  local gpu="$2"
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
    "family": "memory_actor_hidden_shiftgate075_power_refine_probe",
    "task": "Memory",
    "gpu": gpu,
    "output_dir": str(run_dir),
    "started_at": time.time(),
}
(run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
PY
  CUDA_VISIBLE_DEVICES="$gpu" torchrun --standalone --nproc_per_node=1 -m psmn_rl.launch --config "$generated_cfg" &
  LAUNCHED_PID="$!"
}

declare -a FREE_GPUS=()
for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
  FREE_GPUS+=("$gpu")
done

declare -a RUNNING_PIDS=()
declare -A PID_TO_GPU=()
next_cfg_index=0
wait_status=0

while (( next_cfg_index < ${#CONFIGS[@]} )) || (( ${#RUNNING_PIDS[@]} > 0 )); do
  while (( next_cfg_index < ${#CONFIGS[@]} )) && (( ${#RUNNING_PIDS[@]} < GPU_COUNT )); do
    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    launch_config "${CONFIGS[$next_cfg_index]}" "$gpu"
    pid="$LAUNCHED_PID"
    PID_TO_GPU["$pid"]="$gpu"
    RUNNING_PIDS+=("$pid")
    next_cfg_index=$((next_cfg_index + 1))
  done

  if (( ${#RUNNING_PIDS[@]} > 0 )); then
    if ! wait -n; then
      wait_status=$?
    fi
    new_running=()
    for pid in "${RUNNING_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        new_running+=("$pid")
      else
        FREE_GPUS+=("${PID_TO_GPU[$pid]}")
        unset 'PID_TO_GPU[$pid]'
      fi
    done
    RUNNING_PIDS=("${new_running[@]}")
    if (( wait_status != 0 )); then
      break
    fi
  fi
done

if (( wait_status != 0 )); then
  exit "$wait_status"
fi

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
        "run_dir": Path("outputs/experiments/memory_actor_hidden_shiftgate075_power_refine_probe/memory_por_switchy_rerun"),
        "variant": "none",
    },
    {
        "label": "por_actor_hidden_partial_shift22",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_shiftgate075_power_refine_probe/memory_por_switchy_actor_hidden_partial_shift22"),
        "variant": "partial_shift22",
    },
    {
        "label": "por_actor_hidden_partial_shift22_shiftgate075",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_shiftgate075_power_refine_probe/memory_por_switchy_actor_hidden_partial_shift22_shiftgate075"),
        "variant": "shiftgate075",
    },
    {
        "label": "por_actor_hidden_partial_shift22_shiftgate070",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_shiftgate075_power_refine_probe/memory_por_switchy_actor_hidden_partial_shift22_shiftgate070"),
        "variant": "shiftgate070",
    },
    {
        "label": "por_actor_hidden_partial_shift22_shiftgate085",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_shiftgate075_power_refine_probe/memory_por_switchy_actor_hidden_partial_shift22_shiftgate085"),
        "variant": "shiftgate085",
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
            "train/episode_return": float(train_metrics.get("train/episode_return", 0.0)),
            "train/success_rate": float(train_metrics.get("train/success_rate", 0.0)),
            "throughput_fps": float(train_metrics.get("throughput_fps", 0.0)),
            "policy/option_hidden_film_gate_signal_mean": float(train_metrics.get("policy/option_hidden_film_gate_signal_mean", 0.0)),
            "policy/option_hidden_scale_gate_signal_mean": float(train_metrics.get("policy/option_hidden_scale_gate_signal_mean", 0.0)),
            "policy/option_hidden_shift_gate_signal_mean": float(train_metrics.get("policy/option_hidden_shift_gate_signal_mean", 0.0)),
            "policy/option_hidden_film_gate_mean": float(train_metrics.get("policy/option_hidden_film_gate_mean", 0.0)),
            "policy/option_hidden_scale_gate_power": float(train_metrics.get("policy/option_hidden_scale_gate_power", 0.0)),
            "policy/option_hidden_shift_gate_power": float(train_metrics.get("policy/option_hidden_shift_gate_power", 0.0)),
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
shift075 = row("por_actor_hidden_partial_shift22_shiftgate075")
shift070 = row("por_actor_hidden_partial_shift22_shiftgate070")
shift085 = row("por_actor_hidden_partial_shift22_shiftgate085")

lines = [
    "# Memory ShiftGate075 Power-Refine Probe",
    "",
    "- hypothesis: the restored `shiftgate075` local surface may still prefer a nearby additive gate power rather than exactly `0.75`",
    "- task: `MiniGrid-MemoryS9-v0`",
    "- fresh matched runs: `5`",
    "- same-seed architecture isolate: `seed=7` for all runs",
    "- evaluation episodes per mode: `64`",
    "",
    "## Aggregate",
    "",
    "| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Scale Gate Signal | Shift Gate Signal | Gate Mean | Scale Gate Power | Shift Gate Power | Scale Norm | Shift Norm |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for item in (por_base, por_shift22, shift075, shift070, shift085):
    lines.append(
        f"| `{item['label']}` | `{item['greedy/success']:.4f}` | `{item['lower_t005/success']:.4f}` | `{item['gap_t0055/success']:.4f}` | `{item['shoulder_t008/success']:.4f}` | "
        f"`{item['greedy/margin']:.4f}` | `{item['gap_t0055/margin']:.4f}` | `{item['shoulder_t008/margin']:.4f}` | "
        f"`{item['train/episode_return']:.4f}` | `{item['train/success_rate']:.4f}` | `{item['policy/option_hidden_film_gate_signal_mean']:.4f}` | "
        f"`{item['policy/option_hidden_scale_gate_signal_mean']:.4f}` | `{item['policy/option_hidden_shift_gate_signal_mean']:.4f}` | `{item['policy/option_hidden_film_gate_mean']:.4f}` | "
        f"`{item['policy/option_hidden_scale_gate_power']:.2f}` | `{item['policy/option_hidden_shift_gate_power']:.2f}` | `{item['policy/option_hidden_film_scale_norm']:.4f}` | `{item['policy/option_hidden_film_shift_norm']:.4f}` |"
    )

lines.extend(
    [
        "",
        "## Deltas vs ShiftGate075 Incumbent",
        "",
        "| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| `por_actor_hidden_partial_shift22_shiftgate070` | `{shift070['greedy/success'] - shift075['greedy/success']:+.4f}` | `{shift070['lower_t005/success'] - shift075['lower_t005/success']:+.4f}` | `{shift070['gap_t0055/success'] - shift075['gap_t0055/success']:+.4f}` | `{shift070['shoulder_t008/success'] - shift075['shoulder_t008/success']:+.4f}` |",
        f"| `por_actor_hidden_partial_shift22_shiftgate085` | `{shift085['greedy/success'] - shift075['greedy/success']:+.4f}` | `{shift085['lower_t005/success'] - shift075['lower_t005/success']:+.4f}` | `{shift085['gap_t0055/success'] - shift075['gap_t0055/success']:+.4f}` | `{shift085['shoulder_t008/success'] - shift075['shoulder_t008/success']:+.4f}` |",
        "",
    ]
)

interesting = any(
    item["greedy/success"] > 0.0
    and (
        item["greedy/success"] > shift075["greedy/success"]
        or item["lower_t005/success"] > shift075["lower_t005/success"]
        or item["gap_t0055/success"] > shift075["gap_t0055/success"]
        or item["shoulder_t008/success"] > shift075["shoulder_t008/success"]
    )
    for item in (shift070, shift085)
)

if interesting:
    lines.append("- conclusion: at least one nearby shift-gate power preserves nonzero greedy and improves part of the restored `shiftgate075` surface; run a 256-episode confirmation pass before treating it as real")
else:
    lines.append("- conclusion: retuning shift-gate power around `shiftgate075` does not improve the restored local surface")

REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
