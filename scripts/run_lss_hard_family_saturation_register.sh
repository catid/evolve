#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_hard_family_saturation/campaign.yaml}"

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

render_lss_spec() {
  local campaign_config="$1"
  local variant="$2"
  local seed_root="$3"
  local lane="$4"
  local seed="$5"
  local target_path="$6"
  local output_dir="$7"
  source .venv/bin/activate
  python - "$campaign_config" "$variant" "$seed_root" "$lane" "$seed" "$target_path" "$output_dir" <<'PY'
import copy
import sys
from pathlib import Path
import yaml

def deep_merge(base, override):
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)

campaign_path, variant, seed_root, lane, seed, target_path, output_dir = sys.argv[1:]
campaign = yaml.safe_load(Path(campaign_path).read_text()) or {}
student_meta = campaign["students"][variant]
raw = yaml.safe_load(Path(campaign["base_candidate_template"]).read_text()) or {}
seed_root = Path(seed_root)
raw["name"] = f"{lane}_seed_{seed}_{campaign['current_candidate_name']}_{student_meta['output_label']}"
raw["output_dir"] = output_dir
raw["teacher"]["config"] = str(seed_root / "configs" / "flat_dense_ent1e3.yaml")
raw["teacher"]["checkpoint"] = str(seed_root / "flat_dense_ent1e3" / "latest.pt")
raw["student"]["config"] = str(seed_root / "configs" / student_meta["config_name"])
raw["student"]["checkpoint"] = str(seed_root / student_meta["run_name"] / "latest.pt")
deep_merge(raw, {})
target = Path(target_path)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
PY
}

gpu_count() {
  source .venv/bin/activate
  python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
}

run_lss_specs_parallel() {
  local -a specs=("$@")
  local count
  count=$(gpu_count)
  if [[ "${#specs[@]}" -eq 0 ]]; then
    return
  fi
  if [[ "$DEVICE" == "cpu" || "$count" -le 0 ]]; then
    for spec in "${specs[@]}"; do
      source .venv/bin/activate
      python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec" --device "$DEVICE"
    done
    return
  fi
  local slot=0
  local active=0
  local failed=0
  for spec in "${specs[@]}"; do
    local gpu=$((slot % count))
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      source .venv/bin/activate
      python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec" --device cuda
    ) &
    slot=$((slot + 1))
    active=$((active + 1))
    if [[ "$active" -ge "$count" ]]; then
      if ! wait -n; then
        failed=1
      fi
      active=$((active - 1))
    fi
  done
  while [[ "$active" -gt 0 ]]; do
    if ! wait -n; then
      failed=1
    fi
    active=$((active - 1))
  done
  [[ "$failed" -eq 0 ]]
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
readarray -t campaign_values < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
reports = campaign["reports"]
print(campaign["stage_roots"]["stage0_blocks"])
print(campaign["base_configs"]["flat_dense"])
print(campaign["base_configs"]["token_dense"])
print(campaign["base_configs"]["single_expert"])
print(campaign["base_configs"]["sare"])
print(reports["definition"])
print(reports["definition_json"])
print(reports["registration"])
print(reports["baseline_sync"])
print(reports["baseline_csv"])
print(reports["baseline_json"])
print(reports["stage1_shortlist"])
for block in campaign["seed_groups"]["new_block_pool"]["blocks"]:
    print(f"{block['lane']}:{','.join(str(seed) for seed in block['seeds'])}")
PY
)

OUTPUT_ROOT="${campaign_values[0]}"
FLAT_BASE="${campaign_values[1]}"
TOKEN_BASE="${campaign_values[2]}"
SINGLE_BASE="${campaign_values[3]}"
SARE_BASE="${campaign_values[4]}"
DEFINITION_OUTPUT="${campaign_values[5]}"
DEFINITION_JSON="${campaign_values[6]}"
REGISTRATION_OUTPUT="${campaign_values[7]}"
BASELINE_OUTPUT="${campaign_values[8]}"
BASELINE_CSV="${campaign_values[9]}"
BASELINE_JSON="${campaign_values[10]}"
SHORTLIST_OUTPUT="${campaign_values[11]}"
BLOCK_ROWS=("${campaign_values[@]:12}")

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

declare -a lss_specs=()
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

    for variant in token_dense single_expert sare; do
      output_label=$(source .venv/bin/activate && python - "$CAMPAIGN_CONFIG" "$variant" <<'PY'
import sys
from pathlib import Path
import yaml
campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(campaign["students"][sys.argv[2]]["output_label"])
PY
)
      spec_path="${config_root}/${output_label}.yaml"
      output_dir="${seed_root}/${output_label}"
      render_lss_spec "$CAMPAIGN_CONFIG" "$variant" "$seed_root" "$lane" "$seed" "$spec_path" "$output_dir"
      record_command "${output_dir}/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$DEVICE"
      if [[ ! -f "${output_dir}/summary.json" ]]; then
        lss_specs+=("$spec_path")
      fi
    done
  done
done

run_lss_specs_parallel "${lss_specs[@]}"

python -m psmn_rl.analysis.lss_hard_family_saturation definition \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$DEFINITION_OUTPUT" \
  --json-output "$DEFINITION_JSON"

python -m psmn_rl.analysis.lss_hard_family_saturation registration \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REGISTRATION_OUTPUT"

python -m psmn_rl.analysis.lss_hard_family_saturation baseline-sync \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$BASELINE_OUTPUT" \
  --csv "$BASELINE_CSV" \
  --json "$BASELINE_JSON"

python -m psmn_rl.analysis.lss_hard_family_saturation shortlist \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$SHORTLIST_OUTPUT"
