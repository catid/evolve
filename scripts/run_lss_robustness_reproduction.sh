#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
OUTPUT_ROOT="${PSMN_OUTPUT_ROOT:-outputs/reproductions/lss_robustness_baseline}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
SEED_LIST="${PSMN_SEEDS:-7 11 19}"
TEMPLATE="configs/experiments/lss_robustness/lss_baseline_ce.yaml"

mkdir -p "$OUTPUT_ROOT"

if [[ "$DEVICE" == "cpu" ]]; then
  TRAIN_LAUNCHER=(python -m psmn_rl.train)
  ANALYSIS_LAUNCHER=(python -m psmn_rl.analysis.lss_robustness)
else
  NPROC_PER_NODE=$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)
  TRAIN_LAUNCHER=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m psmn_rl.launch)
  ANALYSIS_LAUNCHER=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m psmn_rl.analysis.lss_robustness)
fi

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
  local seed_root="$2"
  local student_run="$3"
  local spec_name="$4"
  local output_dir="$5"
  local target_path="$6"
  python - "$template_path" "$seed_root" "$student_run" "$spec_name" "$output_dir" "$target_path" <<'PY'
import sys
from pathlib import Path
import yaml

template_path, seed_root, student_run, spec_name, output_dir, target_path = sys.argv[1:]
raw = yaml.safe_load(Path(template_path).read_text()) or {}
seed_root = Path(seed_root)
raw["name"] = spec_name
raw["output_dir"] = output_dir
raw["teacher"]["config"] = str(seed_root / "configs" / "flat_dense_ent1e3.yaml")
raw["teacher"]["checkpoint"] = str(seed_root / "flat_dense_ent1e3" / "latest.pt")
raw["student"]["config"] = str(seed_root / "configs" / f"{student_run}.yaml")
raw["student"]["checkpoint"] = str(seed_root / student_run / "latest.pt")
target = Path(target_path)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
PY
}

for seed in ${SEED_LIST}; do
  seed_root="${OUTPUT_ROOT}/seed_${seed}"
  config_root="${seed_root}/configs"
  mkdir -p "$config_root"

  render_train_config \
    "configs/experiments/minigrid_doorkey_flat_dense_ent1e3.yaml" \
    "$seed" \
    "seed_${seed}_flat_dense_ent1e3" \
    "${seed_root}/flat_dense_ent1e3" \
    "${config_root}/flat_dense_ent1e3.yaml"
  record_command "${seed_root}/flat_dense_ent1e3/command.txt" "${TRAIN_LAUNCHER[@]}" --config "${config_root}/flat_dense_ent1e3.yaml" --device "$DEVICE"
  "${TRAIN_LAUNCHER[@]}" --config "${config_root}/flat_dense_ent1e3.yaml" --device "$DEVICE"

  render_train_config \
    "configs/experiments/minigrid_doorkey_token_dense_ent1e3.yaml" \
    "$seed" \
    "seed_${seed}_token_dense_ent1e3" \
    "${seed_root}/token_dense_ent1e3" \
    "${config_root}/token_dense_ent1e3.yaml"
  record_command "${seed_root}/token_dense_ent1e3/command.txt" "${TRAIN_LAUNCHER[@]}" --config "${config_root}/token_dense_ent1e3.yaml" --device "$DEVICE"
  "${TRAIN_LAUNCHER[@]}" --config "${config_root}/token_dense_ent1e3.yaml" --device "$DEVICE"

  render_train_config \
    "configs/experiments/minigrid_doorkey_sare_ent1e3.yaml" \
    "$seed" \
    "seed_${seed}_sare_ent1e3" \
    "${seed_root}/sare_ent1e3" \
    "${config_root}/sare_ent1e3.yaml"
  record_command "${seed_root}/sare_ent1e3/command.txt" "${TRAIN_LAUNCHER[@]}" --config "${config_root}/sare_ent1e3.yaml" --device "$DEVICE"
  "${TRAIN_LAUNCHER[@]}" --config "${config_root}/sare_ent1e3.yaml" --device "$DEVICE"

  render_lss_spec \
    "$TEMPLATE" \
    "$seed_root" \
    "sare_ent1e3" \
    "seed_${seed}_flat_dense_to_sare_lss_baseline" \
    "${seed_root}/flat_dense_to_sare_lss" \
    "${config_root}/flat_dense_to_sare_lss.yaml"
  record_command "${seed_root}/flat_dense_to_sare_lss/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "${config_root}/flat_dense_to_sare_lss.yaml" --device "$DEVICE"
  python -m psmn_rl.analysis.learner_state_supervision run --spec "${config_root}/flat_dense_to_sare_lss.yaml" --device "$DEVICE"
done

"${ANALYSIS_LAUNCHER[@]}" reproduction-note \
  --root "$OUTPUT_ROOT" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output outputs/reports/lss_robustness_reproduction_note.md \
  --csv outputs/reports/lss_robustness_reproduction_note.csv

"${ANALYSIS_LAUNCHER[@]}" heterogeneity-report \
  --root "$OUTPUT_ROOT" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output outputs/reports/lss_seed_heterogeneity_report.md \
  --csv outputs/reports/lss_seed_heterogeneity_report.csv
