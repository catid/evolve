#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

ROOT="outputs/experiments/memory_actor_hidden_option_count_probe"
REPORT_MD="outputs/reports/memory_actor_hidden_option_count_probe.md"
REPORT_JSON="outputs/reports/memory_actor_hidden_option_count_probe.json"
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
  "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_option6.yaml"
  "configs/experiments/minigrid_memory_por_switchy_actor_hidden_partial_shift22_option8.yaml"
)

jobs=()
slot=0
for cfg in "${CONFIGS[@]}"; do
  gpu=$((slot % GPU_COUNT))
  slot=$((slot + 1))
  generated_cfg="$TMP_CFG_DIR/$(basename "$cfg")"
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
    "family": "memory_actor_hidden_option_count_probe",
    "task": "Memory",
    "gpu": gpu,
    "output_dir": str(run_dir),
    "started_at": time.time(),
}
(run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
PY
  CUDA_VISIBLE_DEVICES="$gpu" torchrun --standalone --nproc_per_node=1 -m psmn_rl.launch --config "$generated_cfg" &
  jobs+=("$!")
done

for job in "${jobs[@]}"; do
  wait "$job"
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
        "run_dir": Path("outputs/experiments/memory_actor_hidden_option_count_probe/memory_por_switchy_rerun"),
        "variant": "none",
    },
    {
        "label": "por_actor_hidden_partial_shift22",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_option_count_probe/memory_por_switchy_actor_hidden_partial_shift22"),
        "variant": "option_count_4",
    },
    {
        "label": "por_actor_hidden_partial_shift22_option6",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_option_count_probe/memory_por_switchy_actor_hidden_partial_shift22_option6"),
        "variant": "option_count_6",
    },
    {
        "label": "por_actor_hidden_partial_shift22_option8",
        "run_dir": Path("outputs/experiments/memory_actor_hidden_option_count_probe/memory_por_switchy_actor_hidden_partial_shift22_option8"),
        "variant": "option_count_8",
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
            "option_count": float(config.model.option_count),
            "train/episode_return": float(train_metrics.get("train/episode_return", 0.0)),
            "train/success_rate": float(train_metrics.get("train/success_rate", 0.0)),
            "eval_success_rate": float(train_metrics.get("eval_success_rate", 0.0)),
            "eval_return": float(train_metrics.get("eval_return", 0.0)),
            "throughput_fps": float(train_metrics.get("throughput_fps", 0.0)),
            "policy/option_hidden_film_gate_mean": float(train_metrics.get("policy/option_hidden_film_gate_mean", 0.0)),
            "policy/option_hidden_film_scale_norm": float(train_metrics.get("policy/option_hidden_film_scale_norm", 0.0)),
            "policy/option_hidden_film_shift_norm": float(train_metrics.get("policy/option_hidden_film_shift_norm", 0.0)),
            "option_duration": float(train_metrics.get("option_duration", 0.0)),
            "option_switch_rate": float(train_metrics.get("option_switch_rate", 0.0)),
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
por_option6 = row("por_actor_hidden_partial_shift22_option6")
por_option8 = row("por_actor_hidden_partial_shift22_option8")

lines = [
    "# Memory Actor-Hidden Option-Count Probe",
    "",
    "- hypothesis: the live `partial_shift22` Memory branch may be bottlenecked by too few POR options; a richer option bank could improve greedy or sampled decode while preserving the actor-hidden FiLM surface",
    "- task: `MiniGrid-MemoryS9-v0`",
    "- fresh matched runs: `4`",
    "- same-seed architecture isolate: `seed=7` for all runs",
    "- evaluation episodes per mode: `64`",
    "",
    "## Aggregate",
    "",
    "| Label | Options | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Train Return | Train Success | Throughput | Option Duration | Switch Rate | Gate Mean | Scale Norm | Shift Norm |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for item in (por_base, por_shift22, por_option6, por_option8):
    lines.append(
        f"| `{item['label']}` | `{item['option_count']:.0f}` | `{item['greedy/success']:.4f}` | `{item['lower_t005/success']:.4f}` | `{item['gap_t0055/success']:.4f}` | `{item['shoulder_t008/success']:.4f}` | "
        f"`{item['train/episode_return']:.4f}` | `{item['train/success_rate']:.4f}` | `{item['throughput_fps']:.1f}` | `{item['option_duration']:.4f}` | `{item['option_switch_rate']:.4f}` | "
        f"`{item['policy/option_hidden_film_gate_mean']:.4f}` | `{item['policy/option_hidden_film_scale_norm']:.4f}` | `{item['policy/option_hidden_film_shift_norm']:.4f}` |"
    )

lines.extend(
    [
        "",
        "## Deltas vs Partial-Shift22",
        "",
        "| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| `por_actor_hidden_partial_shift22_option6` | `{por_option6['greedy/success'] - por_shift22['greedy/success']:+.4f}` | `{por_option6['lower_t005/success'] - por_shift22['lower_t005/success']:+.4f}` | `{por_option6['gap_t0055/success'] - por_shift22['gap_t0055/success']:+.4f}` | `{por_option6['shoulder_t008/success'] - por_shift22['shoulder_t008/success']:+.4f}` |",
        f"| `por_actor_hidden_partial_shift22_option8` | `{por_option8['greedy/success'] - por_shift22['greedy/success']:+.4f}` | `{por_option8['lower_t005/success'] - por_shift22['lower_t005/success']:+.4f}` | `{por_option8['gap_t0055/success'] - por_shift22['gap_t0055/success']:+.4f}` | `{por_option8['shoulder_t008/success'] - por_shift22['shoulder_t008/success']:+.4f}` |",
        "",
    ]
)

interesting = any(
    item["greedy/success"] > 0.0
    and (
        item["lower_t005/success"] > por_shift22["lower_t005/success"]
        or item["gap_t0055/success"] > por_shift22["gap_t0055/success"]
        or item["shoulder_t008/success"] > por_shift22["shoulder_t008/success"]
        or item["greedy/success"] > por_shift22["greedy/success"]
    )
    for item in (por_option6, por_option8)
)

if interesting:
    lines.extend(
        [
            "## Outcome",
            "",
            "At least one higher-option variant preserved nonzero greedy conversion while improving part of the sampled band over `partial_shift22`, so this surface should get a longer confirmation pass before being accepted or rejected.",
        ]
    )
else:
    lines.extend(
        [
            "## Outcome",
            "",
            "Increasing POR option count on the `partial_shift22` surface did not produce a better point than the incumbent. Any changed option behavior failed to translate into a greedy-preserving improvement.",
        ]
    )

REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
