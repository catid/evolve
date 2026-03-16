#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
BASELINE_ROOT="${PSMN_BASELINE_ROOT:-outputs/reproductions/lss_robustness_baseline}"
OUTPUT_ROOT="${PSMN_OUTPUT_ROOT:-outputs/experiments/lss_robustness/stage_c}"
TOKEN_SANITY_ROOT="${PSMN_TOKEN_SANITY_ROOT:-outputs/experiments/lss_robustness/token_sanity}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
SEED_LIST="${PSMN_SEEDS:-7 11 19}"
TOKEN_SANITY_SEED="${PSMN_TOKEN_SANITY_SEED:-19}"
TEMPLATE="${PSMN_TEMPLATE:-configs/experiments/lss_robustness/lss_kl_cap_recent_balanced.yaml}"

mkdir -p "$OUTPUT_ROOT" "$TOKEN_SANITY_ROOT"

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
  render_lss_spec \
    "$TEMPLATE" \
    "$seed_root" \
    "sare_ent1e3" \
    "seed_${seed}_improved_lss_sare" \
    "${out_seed_root}/improved_lss_sare" \
    "${config_root}/improved_lss_sare.yaml"
  record_command "${out_seed_root}/improved_lss_sare/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "${config_root}/improved_lss_sare.yaml" --device "$DEVICE"
  python -m psmn_rl.analysis.learner_state_supervision run --spec "${config_root}/improved_lss_sare.yaml" --device "$DEVICE"
done

token_seed_root="${BASELINE_ROOT}/seed_${TOKEN_SANITY_SEED}"
token_config_root="${TOKEN_SANITY_ROOT}/seed_${TOKEN_SANITY_SEED}/configs"
mkdir -p "$token_config_root"
render_lss_spec \
  "$TEMPLATE" \
  "$token_seed_root" \
  "token_dense_ent1e3" \
  "seed_${TOKEN_SANITY_SEED}_improved_lss_token_dense" \
  "${TOKEN_SANITY_ROOT}/seed_${TOKEN_SANITY_SEED}/improved_lss_token_dense" \
  "${token_config_root}/improved_lss_token_dense.yaml"
record_command "${TOKEN_SANITY_ROOT}/seed_${TOKEN_SANITY_SEED}/improved_lss_token_dense/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "${token_config_root}/improved_lss_token_dense.yaml" --device "$DEVICE"
python -m psmn_rl.analysis.learner_state_supervision run --spec "${token_config_root}/improved_lss_token_dense.yaml" --device "$DEVICE"

"${ANALYSIS_LAUNCHER[@]}" sweep-report \
  "$BASELINE_ROOT" \
  "outputs/experiments/lss_robustness/stage_a" \
  "outputs/experiments/lss_robustness/stage_b" \
  "$OUTPUT_ROOT" \
  --token-sanity-root "$TOKEN_SANITY_ROOT" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output outputs/reports/lss_robustness_sweep_report.md \
  --csv outputs/reports/lss_robustness_sweep_report.csv

"${ANALYSIS_LAUNCHER[@]}" multiseed-report \
  --baseline-root "$BASELINE_ROOT" \
  --improved-root "$OUTPUT_ROOT" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output outputs/reports/lss_robustness_multiseed_report.md \
  --csv outputs/reports/lss_robustness_multiseed_report.csv
