#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_canonization_campaign/campaign.yaml}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"

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
  local template="$2"
  local variant="$3"
  local teacher_root="$4"
  local student_root="$5"
  local candidate="$6"
  local lane="$7"
  local seed="$8"
  local target_path="$9"
  local output_dir="${10}"
  source .venv/bin/activate
  python - "$campaign_config" "$template" "$variant" "$teacher_root" "$student_root" "$candidate" "$lane" "$seed" "$target_path" "$output_dir" <<'PY'
import sys
from pathlib import Path
import yaml

campaign_path, template, variant, teacher_root, student_root, candidate, lane, seed, target_path, output_dir = sys.argv[1:]
campaign = yaml.safe_load(Path(campaign_path).read_text()) or {}
student_meta = campaign["students"][variant]
raw = yaml.safe_load(Path(template).read_text()) or {}
teacher_root = Path(teacher_root)
student_root = Path(student_root)
raw["name"] = f"{candidate}_{lane}_seed_{seed}_{student_meta['output_label']}"
raw["output_dir"] = output_dir
raw["teacher"]["config"] = str(teacher_root / "configs" / "flat_dense_ent1e3.yaml")
raw["teacher"]["checkpoint"] = str(teacher_root / "flat_dense_ent1e3" / "latest.pt")
raw["student"]["config"] = str(student_root / "configs" / student_meta["config_name"])
raw["student"]["checkpoint"] = str(student_root / student_meta["run_name"] / "latest.pt")
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
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
reports = campaign["reports"]
print(campaign["base_configs"]["flat_dense"])
print(campaign["base_configs"]["token_dense"])
print(campaign["base_configs"]["single_expert"])
print(campaign["base_configs"]["sare"])
print(campaign["stage_roots"]["stage2_new_block_base"])
print(campaign["stage_roots"]["stage2_candidates"])
print(reports["stage2_report"])
print(reports["stage2_csv"])
print(reports["stage2_json"])
print(campaign["current_candidate_name"])
print(campaign["current_candidate_template"])
print(campaign["existing_lane_roots"]["post_pass_b"]["root"])
print(",".join(str(seed) for seed in campaign["seed_groups"]["hard_block_existing"]["seeds"]))
print(campaign["seed_groups"]["hard_block_existing"]["lane"])
print(campaign["seed_groups"]["hard_block_new"]["lane"])
print(",".join(str(seed) for seed in campaign["seed_groups"]["hard_block_new"]["seeds"]))
for candidate, meta in campaign["candidates"].items():
    print(f"candidate:{candidate}:{meta['template']}")
PY
)

FLAT_BASE="${cfg[0]}"
TOKEN_BASE="${cfg[1]}"
SINGLE_BASE="${cfg[2]}"
SARE_BASE="${cfg[3]}"
NEW_BLOCK_ROOT="${cfg[4]}"
STAGE2_ROOT="${cfg[5]}"
REPORT_OUTPUT="${cfg[6]}"
REPORT_CSV="${cfg[7]}"
REPORT_JSON="${cfg[8]}"
CURRENT_CANDIDATE="${cfg[9]}"
CURRENT_TEMPLATE="${cfg[10]}"
POST_PASS_B_ROOT="${cfg[11]}"
IFS=',' read -r -a EXISTING_SEEDS <<<"${cfg[12]}"
EXISTING_LANE="${cfg[13]}"
NEW_LANE="${cfg[14]}"
IFS=',' read -r -a NEW_SEEDS <<<"${cfg[15]}"
CANDIDATE_ROWS=("${cfg[@]:16}")

mkdir -p "$NEW_BLOCK_ROOT" "$STAGE2_ROOT"

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

for seed in "${NEW_SEEDS[@]}"; do
  seed_root="${NEW_BLOCK_ROOT}/seed_${seed}"
  config_root="${seed_root}/configs"
  mkdir -p "$config_root"

  render_train_config "$FLAT_BASE" "$seed" "${NEW_LANE}_seed_${seed}_flat_dense_ent1e3" "${seed_root}/flat_dense_ent1e3" "${config_root}/flat_dense_ent1e3.yaml"
  record_command "${seed_root}/flat_dense_ent1e3/command.txt" "${TRAIN_LAUNCHER[@]}" --config "${config_root}/flat_dense_ent1e3.yaml" --device "$DEVICE"
  ensure_train_run "${config_root}/flat_dense_ent1e3.yaml" "${seed_root}/flat_dense_ent1e3" "${TRAIN_LAUNCHER[@]}"

  render_train_config "$TOKEN_BASE" "$seed" "${NEW_LANE}_seed_${seed}_token_dense_ent1e3" "${seed_root}/token_dense_ent1e3" "${config_root}/token_dense_ent1e3.yaml"
  record_command "${seed_root}/token_dense_ent1e3/command.txt" "${TRAIN_LAUNCHER[@]}" --config "${config_root}/token_dense_ent1e3.yaml" --device "$DEVICE"
  ensure_train_run "${config_root}/token_dense_ent1e3.yaml" "${seed_root}/token_dense_ent1e3" "${TRAIN_LAUNCHER[@]}"

  render_train_config "$SINGLE_BASE" "$seed" "${NEW_LANE}_seed_${seed}_single_expert_ent1e3" "${seed_root}/single_expert_ent1e3" "${config_root}/single_expert_ent1e3.yaml"
  record_command "${seed_root}/single_expert_ent1e3/command.txt" "${TRAIN_LAUNCHER[@]}" --config "${config_root}/single_expert_ent1e3.yaml" --device "$DEVICE"
  ensure_train_run "${config_root}/single_expert_ent1e3.yaml" "${seed_root}/single_expert_ent1e3" "${TRAIN_LAUNCHER[@]}"

  render_train_config "$SARE_BASE" "$seed" "${NEW_LANE}_seed_${seed}_sare_ent1e3" "${seed_root}/sare_ent1e3" "${config_root}/sare_ent1e3.yaml"
  record_command "${seed_root}/sare_ent1e3/command.txt" "${TRAIN_LAUNCHER[@]}" --config "${config_root}/sare_ent1e3.yaml" --device "$DEVICE"
  ensure_train_run "${config_root}/sare_ent1e3.yaml" "${seed_root}/sare_ent1e3" "${TRAIN_LAUNCHER[@]}"
done

declare -a lss_specs=()
for lane in "$EXISTING_LANE" "$NEW_LANE"; do
  if [[ "$lane" == "$EXISTING_LANE" ]]; then
    root="$POST_PASS_B_ROOT"
    seeds=("${EXISTING_SEEDS[@]}")
  else
    root="$NEW_BLOCK_ROOT"
    seeds=("${NEW_SEEDS[@]}")
  fi

  for seed in "${seeds[@]}"; do
    seed_root="${root}/seed_${seed}"
    config_root="${STAGE2_ROOT}/${CURRENT_CANDIDATE}/${lane}/seed_${seed}/configs"
    mkdir -p "$config_root"
    for variant in sare token_dense single_expert; do
      source .venv/bin/activate
      output_label=$(python - "$CAMPAIGN_CONFIG" "$variant" <<'PY'
import sys
from pathlib import Path
import yaml
campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(campaign["students"][sys.argv[2]]["output_label"])
PY
)
      spec_path="${config_root}/${output_label}.yaml"
      output_dir="${STAGE2_ROOT}/${CURRENT_CANDIDATE}/${lane}/seed_${seed}/${output_label}"
      render_lss_spec "$CAMPAIGN_CONFIG" "$CURRENT_TEMPLATE" "$variant" "$seed_root" "$seed_root" "$CURRENT_CANDIDATE" "$lane" "$seed" "$spec_path" "$output_dir"
      record_command "${output_dir}/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$DEVICE"
      if [[ ! -f "${output_dir}/summary.json" ]]; then
        lss_specs+=("$spec_path")
      fi
    done
  done
done

for row in "${CANDIDATE_ROWS[@]}"; do
  candidate="${row#candidate:}"
  candidate="${candidate%%:*}"
  template="${row#candidate:${candidate}:}"
  for lane in "$EXISTING_LANE" "$NEW_LANE"; do
    if [[ "$lane" == "$EXISTING_LANE" ]]; then
      root="$POST_PASS_B_ROOT"
      seeds=("${EXISTING_SEEDS[@]}")
    else
      root="$NEW_BLOCK_ROOT"
      seeds=("${NEW_SEEDS[@]}")
    fi
    for seed in "${seeds[@]}"; do
      seed_root="${root}/seed_${seed}"
      config_root="${STAGE2_ROOT}/${candidate}/${lane}/seed_${seed}/configs"
      mkdir -p "$config_root"
      spec_path="${config_root}/kl_lss_sare.yaml"
      output_dir="${STAGE2_ROOT}/${candidate}/${lane}/seed_${seed}/kl_lss_sare"
      render_lss_spec "$CAMPAIGN_CONFIG" "$template" "sare" "$seed_root" "$seed_root" "$candidate" "$lane" "$seed" "$spec_path" "$output_dir"
      record_command "${output_dir}/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$DEVICE"
      if [[ ! -f "${output_dir}/summary.json" ]]; then
        lss_specs+=("$spec_path")
      fi
    done
  done
done

run_lss_specs_parallel "${lss_specs[@]}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_canonization_campaign stage2-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage2-root "$STAGE2_ROOT" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
