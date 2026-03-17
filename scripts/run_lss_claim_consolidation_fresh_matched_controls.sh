#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
SEEDS="${PSMN_SEEDS:-23 29 31}"
TEMPLATE="${PSMN_TEMPLATE:-configs/experiments/lss_claim_consolidation/lss_kl.yaml}"
FRESH_BASELINE_ROOT="${PSMN_FRESH_BASELINE_ROOT:-outputs/experiments/lss_claim_hardening/additional_seeds}"
FRESH_SARE_ROOT="${PSMN_FRESH_SARE_ROOT:-outputs/experiments/lss_claim_hardening/additional_seeds}"
FRESH_TOKEN_ROOT="${PSMN_FRESH_TOKEN_ROOT:-outputs/experiments/lss_claim_consolidation/fresh_matched_controls}"
FRESH_REPORT_OUTPUT="${PSMN_FRESH_REPORT_OUTPUT:-outputs/reports/lss_fresh_matched_control_report.md}"
FRESH_REPORT_CSV="${PSMN_FRESH_REPORT_CSV:-outputs/reports/lss_fresh_matched_control_report.csv}"
FRESH_BASELINE_CSV="${PSMN_FRESH_BASELINE_CSV:-outputs/reports/lss_additional_seed_report.csv}"
INTERMEDIATE_REPORT_OUTPUT="${PSMN_INTERMEDIATE_REPORT_OUTPUT:-${FRESH_TOKEN_ROOT}/matched_control_report.md}"
INTERMEDIATE_REPORT_CSV="${PSMN_INTERMEDIATE_REPORT_CSV:-${FRESH_TOKEN_ROOT}/matched_control_report.csv}"
COMBINED_REPORT_OUTPUT="${PSMN_COMBINED_REPORT_OUTPUT:-outputs/reports/lss_combined_doorkey_report.md}"
COMBINED_REPORT_CSV="${PSMN_COMBINED_REPORT_CSV:-outputs/reports/lss_combined_doorkey_report.csv}"
ORIGINAL_BASELINE_ROOT="${PSMN_ORIGINAL_BASELINE_ROOT:-outputs/reproductions/lss_robustness_baseline}"
ORIGINAL_SARE_ROOT="${PSMN_ORIGINAL_SARE_ROOT:-outputs/reproductions/lss_claim_hardening_baseline}"
ORIGINAL_TOKEN_ROOT="${PSMN_ORIGINAL_TOKEN_ROOT:-outputs/experiments/lss_claim_hardening/matched_controls}"
ORIGINAL_MATCHED_CSV="${PSMN_ORIGINAL_MATCHED_CSV:-outputs/reports/lss_matched_control_report.csv}"

PSMN_DEVICE="$DEVICE" \
PSMN_EVAL_EPISODES="$EPISODES" \
PSMN_SEEDS="$SEEDS" \
PSMN_BASELINE_ROOT="$FRESH_BASELINE_ROOT" \
PSMN_SARE_ROOT="$FRESH_SARE_ROOT" \
PSMN_OUTPUT_ROOT="$FRESH_TOKEN_ROOT" \
PSMN_TEMPLATE="$TEMPLATE" \
PSMN_REPORT_OUTPUT="$INTERMEDIATE_REPORT_OUTPUT" \
PSMN_REPORT_CSV="$INTERMEDIATE_REPORT_CSV" \
PSMN_SKIP_REPORT=1 \
bash ./scripts/run_lss_claim_hardening_matched_controls.sh

source .venv/bin/activate
python -m psmn_rl.analysis.lss_claim_consolidation fresh-matched-control-report \
  --baseline-root "$FRESH_BASELINE_ROOT" \
  --sare-root "$FRESH_SARE_ROOT" \
  --token-root "$FRESH_TOKEN_ROOT" \
  --baseline-csv "$FRESH_BASELINE_CSV" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$FRESH_REPORT_OUTPUT" \
  --csv "$FRESH_REPORT_CSV"

python -m psmn_rl.analysis.lss_claim_consolidation combined-doorkey-report \
  --original-baseline-root "$ORIGINAL_BASELINE_ROOT" \
  --original-sare-root "$ORIGINAL_SARE_ROOT" \
  --original-token-root "$ORIGINAL_TOKEN_ROOT" \
  --fresh-baseline-root "$FRESH_BASELINE_ROOT" \
  --fresh-sare-root "$FRESH_SARE_ROOT" \
  --fresh-token-root "$FRESH_TOKEN_ROOT" \
  --original-csv "$ORIGINAL_MATCHED_CSV" \
  --fresh-csv "$FRESH_REPORT_CSV" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$COMBINED_REPORT_OUTPUT" \
  --csv "$COMBINED_REPORT_CSV"
