#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
BASELINE_ROOT="${PSMN_BASELINE_ROOT:-outputs/reproductions/lss_robustness_baseline}"
OUTPUT_ROOT="${PSMN_OUTPUT_ROOT:-outputs/experiments/lss_robustness/stage_b}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
SEED_LIST="${PSMN_SEEDS:-7 19}"
TEMPLATES="${PSMN_TEMPLATES:-configs/experiments/lss_robustness/lss_kl_cap_recent.yaml configs/experiments/lss_robustness/lss_kl_cap_recent_balanced.yaml}"

mkdir -p "$OUTPUT_ROOT"

if [[ "$DEVICE" == "cpu" ]]; then
  ANALYSIS_LAUNCHER=(python -m psmn_rl.analysis.lss_robustness)
else
  NPROC_PER_NODE=$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)
  ANALYSIS_LAUNCHER=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m psmn_rl.analysis.lss_robustness)
fi

record_command() {
  local target="$1"
  shift
  mkdir -p "$(dirname "$target")"
  printf '%q ' "$@" >"$target"
  printf '\n' >>"$target"
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
  seed_root="${BASELINE_ROOT}/seed_${seed}"
  out_seed_root="${OUTPUT_ROOT}/seed_${seed}"
  config_root="${out_seed_root}/configs"
  mkdir -p "$config_root"
  for template in ${TEMPLATES}; do
    method_name="$(basename "$template" .yaml)"
    run_name="flat_dense_to_sare_${method_name}"
    render_lss_spec \
      "$template" \
      "$seed_root" \
      "sare_ent1e3" \
      "seed_${seed}_${run_name}" \
      "${out_seed_root}/${run_name}" \
      "${config_root}/${run_name}.yaml"
    record_command "${out_seed_root}/${run_name}/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "${config_root}/${run_name}.yaml" --device "$DEVICE"
    python -m psmn_rl.analysis.learner_state_supervision run --spec "${config_root}/${run_name}.yaml" --device "$DEVICE"
  done
done

"${ANALYSIS_LAUNCHER[@]}" sweep-report \
  "$BASELINE_ROOT" \
  "outputs/experiments/lss_robustness/stage_a" \
  "$OUTPUT_ROOT" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output outputs/reports/lss_robustness_stage_b_report.md \
  --csv outputs/reports/lss_robustness_stage_b_report.csv
