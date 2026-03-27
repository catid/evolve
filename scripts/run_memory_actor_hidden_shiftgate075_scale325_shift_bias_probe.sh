#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

ROOT="outputs/experiments/memory_actor_hidden_shiftgate075_scale325_shift_bias_probe"
REPORT_MD="outputs/reports/memory_actor_hidden_shiftgate075_scale325_shift_bias_probe.md"
REPORT_JSON="outputs/reports/memory_actor_hidden_shiftgate075_scale325_shift_bias_probe.json"
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
  "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325.yaml"
  "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_shiftbias025.yaml"
  "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_shiftbias05.yaml"
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
    "family": "memory_actor_hidden_shiftgate075_scale325_shift_bias_probe",
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
        "run_dir": Path("outputs/experiments/memory_actor_hidden_shiftgate075_scale325_shift_bias_probe/memory_por_switchy_rerun"),
        "variant": "none",
    },
    {
        "label": "por_actor_hidden_partial_shift22",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_shiftgate075_scale325_shift_bias_probe/memory_por_switchy_actor_hidden_partial_shift22"),
        "variant": "partial_shift22",
    },
    {
        "label": "por_actor_hidden_partial_shift22_shiftgate075_scale325",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_shiftgate075_scale325_shift_bias_probe/memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325"),
        "variant": "shift22_shiftgate075_scale325",
    },
    {
        "label": "por_actor_hidden_partial_shift22_shiftgate075_scale325_shiftbias025",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_shiftgate075_scale325_shift_bias_probe/memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_shiftbias025"),
        "variant": "shift22_shiftgate075_scale325_shiftbias025",
    },
    {
        "label": "por_actor_hidden_partial_shift22_shiftgate075_scale325_shiftbias05",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_shiftgate075_scale325_shift_bias_probe/memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325_shiftbias05"),
        "variant": "shift22_shiftgate075_scale325_shiftbias05",
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
            "policy/option_hidden_gate_bias": float(train_metrics.get("policy/option_hidden_gate_bias", 0.0)),
            "policy/option_hidden_shift_gate_bias_scale": float(train_metrics.get("policy/option_hidden_shift_gate_bias_scale", 0.0)),
            "policy/option_hidden_gate_bias_mean": float(train_metrics.get("policy/option_hidden_gate_bias_mean", 0.0)),
            "policy/option_hidden_shift_gate_bias_mean": float(train_metrics.get("policy/option_hidden_shift_gate_bias_mean", 0.0)),
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
shift22 = row("por_actor_hidden_partial_shift22")
winner = row("por_actor_hidden_partial_shift22_shiftgate075_scale325")
bias025 = row("por_actor_hidden_partial_shift22_shiftgate075_scale325_shiftbias025")
bias05 = row("por_actor_hidden_partial_shift22_shiftgate075_scale325_shiftbias05")

lines = [
    "# Memory ShiftGate075 Scale325 Shift-Bias Probe",
    "",
    "- hypothesis: a very small learned positive bias on the shift gate may recover a bit of low-gate headroom on `shiftgate075_scale325` without disturbing the restored local plateau",
    "- task: `MiniGrid-MemoryS9-v0`",
    "- fresh matched runs: `5`",
    "- same-seed architecture isolate: `seed=7` for all runs",
    "- evaluation episodes per mode: `64`",
    "",
    "## Aggregate",
    "",
    "| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Train Return | Train Success | Gate Bias | Shift Bias Scale | Gate Bias Mean | Shift Bias Mean | Gate Mean | Scale Norm | Shift Norm |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for item in [por_base, shift22, winner, bias025, bias05]:
    lines.append(
        "| `{label}` | `{greedy:.4f}` | `{lower:.4f}` | `{gap:.4f}` | `{shoulder:.4f}` | `{train_return:.4f}` | `{train_success:.4f}` | `{gate_bias:.0f}` | `{shift_bias_scale:.3f}` | `{gate_bias_mean:.4f}` | `{shift_bias_mean:.4f}` | `{gate_mean:.4f}` | `{scale_norm:.4f}` | `{shift_norm:.4f}` |".format(
            label=item["label"],
            greedy=item["greedy/success"],
            lower=item["lower_t005/success"],
            gap=item["gap_t0055/success"],
            shoulder=item["shoulder_t008/success"],
            train_return=item["train/episode_return"],
            train_success=item["train/success_rate"],
            gate_bias=item["policy/option_hidden_gate_bias"],
            shift_bias_scale=item["policy/option_hidden_shift_gate_bias_scale"],
            gate_bias_mean=item["policy/option_hidden_gate_bias_mean"],
            shift_bias_mean=item["policy/option_hidden_shift_gate_bias_mean"],
            gate_mean=item["policy/option_hidden_film_gate_mean"],
            scale_norm=item["policy/option_hidden_film_scale_norm"],
            shift_norm=item["policy/option_hidden_film_shift_norm"],
        )
    )

lines.extend(
    [
        "",
        "## Deltas vs ShiftGate075 Scale325",
        "",
        "- `shiftbias025`: greedy {g1:+.4f}, lower {l1:+.4f}, gap {gp1:+.4f}, shoulder {s1:+.4f}".format(
            g1=bias025["greedy/success"] - winner["greedy/success"],
            l1=bias025["lower_t005/success"] - winner["lower_t005/success"],
            gp1=bias025["gap_t0055/success"] - winner["gap_t0055/success"],
            s1=bias025["shoulder_t008/success"] - winner["shoulder_t008/success"],
        ),
        "- `shiftbias05`: greedy {g2:+.4f}, lower {l2:+.4f}, gap {gp2:+.4f}, shoulder {s2:+.4f}".format(
            g2=bias05["greedy/success"] - winner["greedy/success"],
            l2=bias05["lower_t005/success"] - winner["lower_t005/success"],
            gp2=bias05["gap_t0055/success"] - winner["gap_t0055/success"],
            s2=bias05["shoulder_t008/success"] - winner["shoulder_t008/success"],
        ),
        "",
        "## Conclusion",
        "",
    ]
)

survivors = []
for item in [bias025, bias05]:
    if item["greedy/success"] >= winner["greedy/success"] and item["gap_t0055/success"] >= winner["gap_t0055/success"]:
        survivors.append(item["label"])

if survivors:
    lines.append(
        "- survivors worth confirmation: {}".format(", ".join(f"`{label}`" for label in survivors))
    )
else:
    lines.append(
        "- micro shift-gate biases around `shiftgate075_scale325` do not improve the current local surface enough to justify a 256-episode confirmation pass"
    )

REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
