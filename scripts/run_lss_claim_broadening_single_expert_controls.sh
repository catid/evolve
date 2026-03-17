#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
SEED_LIST="${PSMN_SEEDS:-7 11 19}"
BASELINE_ROOT="${PSMN_BASELINE_ROOT:-outputs/reproductions/lss_robustness_baseline}"
OUTPUT_ROOT="${PSMN_OUTPUT_ROOT:-outputs/experiments/lss_claim_broadening/single_expert_controls}"
TEMPLATE="${PSMN_TEMPLATE:-configs/experiments/lss_claim_broadening/lss_kl.yaml}"
BASE_CONFIG="${PSMN_BASE_CONFIG:-configs/experiments/minigrid_doorkey_single_expert_ent1e3.yaml}"
REPORT_OUTPUT="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_single_expert_matched_control_report.md}"
REPORT_CSV="${PSMN_REPORT_CSV:-outputs/reports/lss_single_expert_matched_control_report.csv}"
ORIGINAL_MATCHED_CSV="${PSMN_ORIGINAL_MATCHED_CSV:-outputs/reports/lss_matched_control_report.csv}"
SKIP_REPORT="${PSMN_SKIP_REPORT:-0}"

mkdir -p "$OUTPUT_ROOT"

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
Path(target_path).parent.mkdir(parents=True, exist_ok=True)
Path(target_path).write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
PY
}

render_lss_spec() {
  local template_path="$1"
  local teacher_seed_root="$2"
  local student_seed_root="$3"
  local spec_name="$4"
  local output_dir="$5"
  local target_path="$6"
  source .venv/bin/activate
  python - "$template_path" "$teacher_seed_root" "$student_seed_root" "$spec_name" "$output_dir" "$target_path" <<'PY'
import sys
from pathlib import Path
import yaml

template_path, teacher_seed_root, student_seed_root, spec_name, output_dir, target_path = sys.argv[1:]
raw = yaml.safe_load(Path(template_path).read_text()) or {}
teacher_seed_root = Path(teacher_seed_root)
student_seed_root = Path(student_seed_root)
raw["name"] = spec_name
raw["output_dir"] = output_dir
raw["teacher"]["config"] = str(teacher_seed_root / "configs" / "flat_dense_ent1e3.yaml")
raw["teacher"]["checkpoint"] = str(teacher_seed_root / "flat_dense_ent1e3" / "latest.pt")
raw["student"]["config"] = str(student_seed_root / "configs" / "single_expert_ent1e3.yaml")
raw["student"]["checkpoint"] = str(student_seed_root / "single_expert_ent1e3" / "latest.pt")
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
  if [[ "$DEVICE" == "cpu" || "$count" -le 0 ]]; then
    for spec in "${specs[@]}"; do
      source .venv/bin/activate
      python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec" --device "${DEVICE}"
    done
    return
  fi
  local slot=0
  local -a pids=()
  for spec in "${specs[@]}"; do
    local gpu=$((slot % count))
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      source .venv/bin/activate
      python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec" --device cuda
    ) &
    pids+=("$!")
    slot=$((slot + 1))
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    return 1
  fi
}

if [[ "$DEVICE" == "cpu" ]]; then
  TRAIN_LAUNCHER=(python -m psmn_rl.train)
else
  source .venv/bin/activate
  NPROC_PER_NODE=$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)
  TRAIN_LAUNCHER=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m psmn_rl.launch)
fi

declare -a specs=()
for seed in ${SEED_LIST}; do
  teacher_seed_root="${BASELINE_ROOT}/seed_${seed}"
  student_seed_root="${OUTPUT_ROOT}/seed_${seed}"
  config_root="${student_seed_root}/configs"
  mkdir -p "$config_root"

  render_train_config \
    "$BASE_CONFIG" \
    "$seed" \
    "seed_${seed}_single_expert_ent1e3" \
    "${student_seed_root}/single_expert_ent1e3" \
    "${config_root}/single_expert_ent1e3.yaml"

  record_command "${student_seed_root}/single_expert_ent1e3/command.txt" "${TRAIN_LAUNCHER[@]}" --config "${config_root}/single_expert_ent1e3.yaml" --device "$DEVICE"
  "${TRAIN_LAUNCHER[@]}" --config "${config_root}/single_expert_ent1e3.yaml" --device "$DEVICE"

  spec_path="${config_root}/kl_lss_single_expert.yaml"
  render_lss_spec \
    "$TEMPLATE" \
    "$teacher_seed_root" \
    "$student_seed_root" \
    "seed_${seed}_kl_lss_single_expert" \
    "${student_seed_root}/kl_lss_single_expert" \
    "$spec_path"
  record_command "${student_seed_root}/kl_lss_single_expert/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$DEVICE"
  specs+=("$spec_path")
done

run_lss_specs_parallel "${specs[@]}"

if [[ "$SKIP_REPORT" != "1" ]]; then
  source .venv/bin/activate
  python -m psmn_rl.analysis.lss_claim_broadening single-expert-matched-control-report \
    --original-csv "$ORIGINAL_MATCHED_CSV" \
    --single-expert-root "$OUTPUT_ROOT" \
    --episodes "$EPISODES" \
    --device "$DEVICE" \
    --output "$REPORT_OUTPUT" \
    --csv "$REPORT_CSV"
fi
