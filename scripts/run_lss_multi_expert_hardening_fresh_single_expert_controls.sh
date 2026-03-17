#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
TEMPLATE="${PSMN_TEMPLATE:-configs/experiments/lss_multi_expert_hardening/lss_kl.yaml}"
FRESH_ROOT="${PSMN_FRESH_OUTPUT_ROOT:-outputs/experiments/lss_multi_expert_hardening/fresh_single_expert_controls}"
FRESH_EXTRA_ROOT="${PSMN_FRESH_EXTRA_OUTPUT_ROOT:-outputs/experiments/lss_multi_expert_hardening/fresh_extra_single_expert_controls}"
OUTPUT_PATH="${PSMN_REPORT_OUTPUT:-outputs/reports/lss_fresh_single_expert_matched_control_report.md}"
CSV_PATH="${PSMN_REPORT_CSV:-outputs/reports/lss_fresh_single_expert_matched_control_report.csv}"

PSMN_DEVICE="$DEVICE" \
PSMN_EVAL_EPISODES="$EPISODES" \
PSMN_SEEDS="23 29 31" \
PSMN_BASELINE_ROOT="outputs/experiments/lss_claim_hardening/additional_seeds" \
PSMN_OUTPUT_ROOT="$FRESH_ROOT" \
PSMN_TEMPLATE="$TEMPLATE" \
PSMN_SKIP_REPORT=1 \
bash ./scripts/run_lss_claim_broadening_single_expert_controls.sh

PSMN_DEVICE="$DEVICE" \
PSMN_EVAL_EPISODES="$EPISODES" \
PSMN_SEEDS="37 41 43" \
PSMN_BASELINE_ROOT="outputs/experiments/lss_claim_broadening/additional_fresh_block" \
PSMN_OUTPUT_ROOT="$FRESH_EXTRA_ROOT" \
PSMN_TEMPLATE="$TEMPLATE" \
PSMN_SKIP_REPORT=1 \
bash ./scripts/run_lss_claim_broadening_single_expert_controls.sh

source .venv/bin/activate
python -m psmn_rl.analysis.lss_multi_expert_hardening fresh-single-expert-matched-control-report \
  --original-csv outputs/reports/lss_single_expert_matched_control_report.csv \
  --fresh-csv outputs/reports/lss_fresh_matched_control_report.csv \
  --fresh-extra-csv outputs/reports/lss_additional_fresh_seed_block_report.csv \
  --fresh-single-expert-root "$FRESH_ROOT" \
  --fresh-extra-single-expert-root "$FRESH_EXTRA_ROOT" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$OUTPUT_PATH" \
  --csv "$CSV_PATH"
