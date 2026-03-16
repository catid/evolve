#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
OUTPUT_ROOT="${PSMN_OUTPUT_ROOT:-outputs/experiments/greedy_recovery/self_imitation}"
SUCCESS_TARGET="${PSMN_SUCCESS_TARGET:-64}"
MAX_EPISODES="${PSMN_MAX_EPISODES:-256}"
EVAL_EPISODES="${PSMN_EVAL_EPISODES:-32}"
EPOCHS="${PSMN_EPOCHS:-8}"
BATCH_SIZE="${PSMN_BATCH_SIZE:-128}"
LEARNING_RATE="${PSMN_LEARNING_RATE:-0.0001}"

mkdir -p "$OUTPUT_ROOT"

GPU_COUNT=0
if [[ "$DEVICE" != "cpu" ]]; then
  GPU_COUNT=$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
fi

declare -a RUNS=(
  "single_expert|outputs/reproductions/greedy_recovery_baseline/minigrid_doorkey_single_expert_ent1e3/latest.pt|configs/experiments/minigrid_doorkey_single_expert_ent1e3.yaml"
  "sare|outputs/reproductions/greedy_recovery_baseline/minigrid_doorkey_sare_ent1e3/latest.pt|configs/experiments/minigrid_doorkey_sare_ent1e3.yaml"
)

declare -a TARGETS=(
  "policy_head"
  "policy_head_plus_last_shared"
)

declare -a WEIGHTINGS=(
  "uniform"
  "return"
)

launch_job() {
  local checkpoint="$1"
  local config="$2"
  local run_name="$3"
  local target="$4"
  local weighting="$5"
  python -m psmn_rl.analysis.self_imitation run \
    --checkpoint "$checkpoint" \
    --config "$config" \
    --output-dir "$OUTPUT_ROOT/$run_name" \
    --device "$DEVICE" \
    --teacher-temperature 1.0 \
    --success-target "$SUCCESS_TARGET" \
    --max-episodes "$MAX_EPISODES" \
    --target "$target" \
    --weighting "$weighting" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --eval-episodes "$EVAL_EPISODES"
}

active_jobs=0
job_index=0
for run in "${RUNS[@]}"; do
  IFS="|" read -r variant checkpoint config <<<"$run"
  for target in "${TARGETS[@]}"; do
    for weighting in "${WEIGHTINGS[@]}"; do
      run_name="${variant}_${target}_${weighting}"
      if (( GPU_COUNT > 1 )); then
        gpu_id=$(( job_index % GPU_COUNT ))
        CUDA_VISIBLE_DEVICES="$gpu_id" launch_job "$checkpoint" "$config" "$run_name" "$target" "$weighting" &
        active_jobs=$((active_jobs + 1))
        job_index=$((job_index + 1))
        if (( active_jobs >= GPU_COUNT )); then
          wait -n
          active_jobs=$((active_jobs - 1))
        fi
      else
        launch_job "$checkpoint" "$config" "$run_name" "$target" "$weighting"
      fi
    done
  done
done

if (( active_jobs > 0 )); then
  wait
fi

python -m psmn_rl.analysis.self_imitation report \
  "$OUTPUT_ROOT" \
  --output outputs/reports/self_imitation_report.md \
  --csv outputs/reports/self_imitation_report.csv
