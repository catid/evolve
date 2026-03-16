#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
OUTPUT_ROOT="${PSMN_OUTPUT_ROOT:-outputs/experiments/teacher_extraction/multiseed}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
SEED_LIST="${PSMN_SEEDS:-7 11 19}"
CONFIG_ROOT="${OUTPUT_ROOT}/configs"

mkdir -p "$CONFIG_ROOT"

if [[ "$DEVICE" == "cpu" ]]; then
  TRAIN_LAUNCHER=(python -m psmn_rl.train)
else
  NPROC_PER_NODE=$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)
  TRAIN_LAUNCHER=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m psmn_rl.launch)
fi

render_train_config() {
  local base_config="$1"
  local seed="$2"
  local run_name="$3"
  local output_dir="$4"
  local target_path="$5"
  python - "$base_config" "$seed" "$run_name" "$output_dir" "$target_path" <<'PY'
import sys
from pathlib import Path
import yaml

base_config, seed, run_name, output_dir, target_path = sys.argv[1:]
raw = yaml.safe_load(Path(base_config).read_text()) or {}
raw["seed"] = int(seed)
raw.setdefault("logging", {})
raw["logging"]["run_name"] = run_name
raw["logging"]["output_dir"] = output_dir
Path(target_path).parent.mkdir(parents=True, exist_ok=True)
Path(target_path).write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
PY
}

render_lss_spec() {
  local seed="$1"
  local seed_root="$2"
  local target_path="$3"
  python - "$seed" "$seed_root" "$target_path" <<'PY'
import sys
from pathlib import Path
import yaml

seed, seed_root, target_path = sys.argv[1:]
seed_root = Path(seed_root)
spec = {
    "name": f"seed_{seed}_flat_dense_to_sare_lss",
    "output_dir": str(seed_root / "flat_dense_to_sare_lss"),
    "teacher": {
        "config": str(seed_root / "configs" / "flat_dense_ent1e3.yaml"),
        "checkpoint": str(seed_root / "flat_dense_ent1e3" / "latest.pt"),
        "greedy": True,
        "temperature": 1.0,
    },
    "student": {
        "config": str(seed_root / "configs" / "sare_ent1e3.yaml"),
        "checkpoint": str(seed_root / "sare_ent1e3" / "latest.pt"),
        "target": "policy_head_plus_last_shared",
        "loss": "ce",
        "weighting": "uniform",
        "learning_rate": 1e-4,
        "batch_size": 128,
        "epochs": 4,
    },
    "loop": {
        "rounds": 4,
        "episodes_per_round": 64,
        "max_episodes_per_round": 96,
    },
    "evaluation": {
        "episodes": 64,
    },
}
target = Path(target_path)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
PY
}

for seed in ${SEED_LIST}; do
  seed_root="${OUTPUT_ROOT}/seed_${seed}"
  seed_config_root="${seed_root}/configs"
  mkdir -p "$seed_config_root"

  render_train_config \
    "configs/experiments/minigrid_doorkey_flat_dense_ent1e3.yaml" \
    "$seed" \
    "seed_${seed}_flat_dense_ent1e3" \
    "${seed_root}/flat_dense_ent1e3" \
    "${seed_config_root}/flat_dense_ent1e3.yaml"

  render_train_config \
    "configs/experiments/minigrid_doorkey_token_dense_ent1e3.yaml" \
    "$seed" \
    "seed_${seed}_token_dense_ent1e3" \
    "${seed_root}/token_dense_ent1e3" \
    "${seed_config_root}/token_dense_ent1e3.yaml"

  render_train_config \
    "configs/experiments/minigrid_doorkey_sare_ent1e3.yaml" \
    "$seed" \
    "seed_${seed}_sare_ent1e3" \
    "${seed_root}/sare_ent1e3" \
    "${seed_config_root}/sare_ent1e3.yaml"

  "${TRAIN_LAUNCHER[@]}" --config "${seed_config_root}/flat_dense_ent1e3.yaml" --device "$DEVICE"
  "${TRAIN_LAUNCHER[@]}" --config "${seed_config_root}/token_dense_ent1e3.yaml" --device "$DEVICE"
  "${TRAIN_LAUNCHER[@]}" --config "${seed_config_root}/sare_ent1e3.yaml" --device "$DEVICE"

  render_lss_spec "$seed" "$seed_root" "${seed_config_root}/flat_dense_to_sare_lss.yaml"
  python -m psmn_rl.analysis.learner_state_supervision run \
    --spec "${seed_config_root}/flat_dense_to_sare_lss.yaml" \
    --device "$DEVICE"
done

paths=()
for seed in ${SEED_LIST}; do
  paths+=("${OUTPUT_ROOT}/seed_${seed}/token_dense_ent1e3")
  paths+=("${OUTPUT_ROOT}/seed_${seed}/flat_dense_to_sare_lss")
done

python -m psmn_rl.analysis.policy_diagnostics \
  "${paths[@]}" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --group-by run_name \
  --output outputs/reports/teacher_extraction_multiseed_report.md \
  --csv outputs/reports/teacher_extraction_multiseed_report.csv
