#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_successor_validation/campaign.yaml}"

record_command() {
  local target="$1"
  shift
  mkdir -p "$(dirname "$target")"
  printf '%q ' "$@" >"$target"
  printf '\n' >>"$target"
}

render_train_config() {
  local base_config="$1"
  local seed="$2"
  local run_name="$3"
  local output_dir="$4"
  local target_path="$5"
  source .venv/bin/activate
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
target = Path(target_path)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
PY
}

ensure_train_run() {
  local config_path="$1"
  local output_dir="$2"
  shift 2
  local -a launcher=("$@")
  if [[ -f "${output_dir}/latest.pt" ]]; then
    return
  fi
  "${launcher[@]}" --config "$config_path" --device "$DEVICE"
}

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(campaign["stage_roots"]["stage0_blocks"])
print(campaign["base_configs"]["flat_dense"])
print(campaign["base_configs"]["token_dense"])
print(campaign["base_configs"]["single_expert"])
print(campaign["base_configs"]["sare"])
print(campaign["reports"]["registration"])
for block in campaign["fresh_blocks"]["blocks"]:
    print(f"{block['lane']}:{','.join(str(seed) for seed in block['seeds'])}")
PY
)

OUTPUT_ROOT="${cfg[0]}"
FLAT_BASE="${cfg[1]}"
TOKEN_BASE="${cfg[2]}"
SINGLE_BASE="${cfg[3]}"
SARE_BASE="${cfg[4]}"
REGISTRATION_OUTPUT="${cfg[5]}"
BLOCK_ROWS=("${cfg[@]:6}")

mkdir -p "$OUTPUT_ROOT"

if [[ "$DEVICE" == "cpu" ]]; then
  TRAIN_LAUNCHER=(python -m psmn_rl.train)
else
  NPROC_PER_NODE=$(source .venv/bin/activate && python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)
  TRAIN_LAUNCHER=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m psmn_rl.launch)
fi

for block_row in "${BLOCK_ROWS[@]}"; do
  lane="${block_row%%:*}"
  seeds_csv="${block_row#*:}"
  IFS=',' read -r -a seeds <<<"$seeds_csv"
  for seed in "${seeds[@]}"; do
    seed_root="${OUTPUT_ROOT}/${lane}/seed_${seed}"
    config_root="${seed_root}/configs"
    mkdir -p "$config_root"

    render_train_config "$FLAT_BASE" "$seed" "${lane}_seed_${seed}_flat_dense_ent1e3" "${seed_root}/flat_dense_ent1e3" "${config_root}/flat_dense_ent1e3.yaml"
    record_command "${seed_root}/flat_dense_ent1e3/command.txt" "${TRAIN_LAUNCHER[@]}" --config "${config_root}/flat_dense_ent1e3.yaml" --device "$DEVICE"
    ensure_train_run "${config_root}/flat_dense_ent1e3.yaml" "${seed_root}/flat_dense_ent1e3" "${TRAIN_LAUNCHER[@]}"

    render_train_config "$TOKEN_BASE" "$seed" "${lane}_seed_${seed}_token_dense_ent1e3" "${seed_root}/token_dense_ent1e3" "${config_root}/token_dense_ent1e3.yaml"
    record_command "${seed_root}/token_dense_ent1e3/command.txt" "${TRAIN_LAUNCHER[@]}" --config "${config_root}/token_dense_ent1e3.yaml" --device "$DEVICE"
    ensure_train_run "${config_root}/token_dense_ent1e3.yaml" "${seed_root}/token_dense_ent1e3" "${TRAIN_LAUNCHER[@]}"

    render_train_config "$SINGLE_BASE" "$seed" "${lane}_seed_${seed}_single_expert_ent1e3" "${seed_root}/single_expert_ent1e3" "${config_root}/single_expert_ent1e3.yaml"
    record_command "${seed_root}/single_expert_ent1e3/command.txt" "${TRAIN_LAUNCHER[@]}" --config "${config_root}/single_expert_ent1e3.yaml" --device "$DEVICE"
    ensure_train_run "${config_root}/single_expert_ent1e3.yaml" "${seed_root}/single_expert_ent1e3" "${TRAIN_LAUNCHER[@]}"

    render_train_config "$SARE_BASE" "$seed" "${lane}_seed_${seed}_sare_ent1e3" "${seed_root}/sare_ent1e3" "${config_root}/sare_ent1e3.yaml"
    record_command "${seed_root}/sare_ent1e3/command.txt" "${TRAIN_LAUNCHER[@]}" --config "${config_root}/sare_ent1e3.yaml" --device "$DEVICE"
    ensure_train_run "${config_root}/sare_ent1e3.yaml" "${seed_root}/sare_ent1e3" "${TRAIN_LAUNCHER[@]}"
  done
done

source .venv/bin/activate
python -m psmn_rl.analysis.lss_successor_validation registration \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REGISTRATION_OUTPUT"
