#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
SEEDS="${PSMN_SEEDS:-37 41 43}"
TEMPLATE="${PSMN_TEMPLATE:-configs/experiments/lss_claim_broadening/lss_kl.yaml}"
BASELINE_ROOT="${PSMN_BASELINE_ROOT:-outputs/experiments/lss_claim_broadening/additional_fresh_block}"
TOKEN_ROOT="${PSMN_TOKEN_ROOT:-outputs/experiments/lss_claim_broadening/additional_fresh_matched_controls}"
BASELINE_REPORT_OUTPUT="${PSMN_BASELINE_REPORT_OUTPUT:-outputs/reports/lss_additional_fresh_baseline_report.md}"
BASELINE_REPORT_CSV="${PSMN_BASELINE_REPORT_CSV:-outputs/reports/lss_additional_fresh_baseline_report.csv}"
BLOCK_REPORT_OUTPUT="${PSMN_BLOCK_REPORT_OUTPUT:-outputs/reports/lss_additional_fresh_seed_block_report.md}"
BLOCK_REPORT_CSV="${PSMN_BLOCK_REPORT_CSV:-outputs/reports/lss_additional_fresh_seed_block_report.csv}"
COMBINED_REPORT_OUTPUT="${PSMN_COMBINED_REPORT_OUTPUT:-outputs/reports/lss_expanded_combined_doorkey_report.md}"
COMBINED_REPORT_CSV="${PSMN_COMBINED_REPORT_CSV:-outputs/reports/lss_expanded_combined_doorkey_report.csv}"
ORIGINAL_MATCHED_CSV="${PSMN_ORIGINAL_MATCHED_CSV:-outputs/reports/lss_matched_control_report.csv}"
FRESH_MATCHED_CSV="${PSMN_FRESH_MATCHED_CSV:-outputs/reports/lss_fresh_matched_control_report.csv}"
SINGLE_EXPERT_CSV="${PSMN_SINGLE_EXPERT_CSV:-outputs/reports/lss_single_expert_matched_control_report.csv}"

PSMN_DEVICE="$DEVICE" \
PSMN_EVAL_EPISODES="$EPISODES" \
PSMN_SEEDS="$SEEDS" \
PSMN_OUTPUT_ROOT="$BASELINE_ROOT" \
PSMN_TEMPLATE="$TEMPLATE" \
PSMN_REPORT_OUTPUT="$BASELINE_REPORT_OUTPUT" \
PSMN_REPORT_CSV="$BASELINE_REPORT_CSV" \
bash ./scripts/run_lss_claim_hardening_additional_seeds.sh

PSMN_DEVICE="$DEVICE" \
PSMN_EVAL_EPISODES="$EPISODES" \
PSMN_SEEDS="$SEEDS" \
PSMN_BASELINE_ROOT="$BASELINE_ROOT" \
PSMN_SARE_ROOT="$BASELINE_ROOT" \
PSMN_OUTPUT_ROOT="$TOKEN_ROOT" \
PSMN_TEMPLATE="$TEMPLATE" \
PSMN_SKIP_REPORT=1 \
bash ./scripts/run_lss_claim_hardening_matched_controls.sh

source .venv/bin/activate
python -m psmn_rl.analysis.lss_claim_broadening additional-fresh-seed-block-report \
  --baseline-csv "$BASELINE_REPORT_CSV" \
  --token-root "$TOKEN_ROOT" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$BLOCK_REPORT_OUTPUT" \
  --csv "$BLOCK_REPORT_CSV"

COMBINED_ARGS=(
  -m psmn_rl.analysis.lss_claim_broadening
  expanded-combined-doorkey-report
  --original-csv "$ORIGINAL_MATCHED_CSV"
  --fresh-csv "$FRESH_MATCHED_CSV"
  --extra-csv "$BLOCK_REPORT_CSV"
  --episodes "$EPISODES"
  --output "$COMBINED_REPORT_OUTPUT"
  --csv "$COMBINED_REPORT_CSV"
)
if [[ -f "$SINGLE_EXPERT_CSV" ]]; then
  COMBINED_ARGS+=(--single-expert-csv "$SINGLE_EXPERT_CSV")
fi
python "${COMBINED_ARGS[@]}"
