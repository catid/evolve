#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
TEMPLATE="${PSMN_TEMPLATE:-configs/experiments/lss_frozen_claim/lss_kl.yaml}"
OUTPUT_ROOT="${PSMN_OUTPUT_ROOT:-outputs/experiments/lss_frozen_claim/final_block_single_expert_controls}"
BASELINE_ROOT="${PSMN_BASELINE_ROOT:-outputs/experiments/lss_multi_expert_hardening/final_fresh_block}"
BASELINE_CSV="${PSMN_BASELINE_CSV:-outputs/reports/lss_final_fresh_seed_block_report.csv}"
OUTPUT_PATH="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_final_block_single_expert_control_report.md}"
CSV_PATH="${PSMN_REPORT_CSV:-outputs/reports/lss_final_block_single_expert_control_report.csv}"

PSMN_DEVICE="$DEVICE" \
PSMN_EVAL_EPISODES="$EPISODES" \
PSMN_SEEDS="47 53 59" \
PSMN_BASELINE_ROOT="$BASELINE_ROOT" \
PSMN_OUTPUT_ROOT="$OUTPUT_ROOT" \
PSMN_TEMPLATE="$TEMPLATE" \
PSMN_SKIP_REPORT=1 \
bash ./scripts/run_lss_claim_broadening_single_expert_controls.sh

source .venv/bin/activate
python -m psmn_rl.analysis.lss_frozen_claim final-block-single-expert-control-report \
  --baseline-csv "$BASELINE_CSV" \
  --single-expert-root "$OUTPUT_ROOT" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$OUTPUT_PATH" \
  --csv "$CSV_PATH"
