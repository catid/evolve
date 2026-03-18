#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_long_campaign/campaign.yaml}"
COMBINED_CSV="${PSMN_COMBINED_CSV:-outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.csv}"
FINAL_CSV="${PSMN_FINAL_CSV:-outputs/reports/lss_final_block_single_expert_control_report.csv}"
REGISTRATION_OUTPUT="${PSMN_REGISTRATION_OUTPUT:-outputs/reports/long_campaign_registration.md}"
BASELINE_OUTPUT="${PSMN_BASELINE_OUTPUT:-outputs/reports/long_campaign_baseline_sync.md}"
SHORTLIST_OUTPUT="${PSMN_SHORTLIST_OUTPUT:-outputs/reports/long_campaign_mechanism_shortlist.md}"
FROZEN_VALIDATION_OUTPUT="${PSMN_LONG_CAMPAIGN_FROZEN_VALIDATION_OUTPUT:-outputs/reports/long_campaign_frozen_baseline_validation.md}"
FROZEN_VALIDATION_CSV="${PSMN_LONG_CAMPAIGN_FROZEN_VALIDATION_CSV:-outputs/reports/long_campaign_frozen_baseline_validation.csv}"
FROZEN_VALIDATION_JSON="${PSMN_LONG_CAMPAIGN_FROZEN_VALIDATION_JSON:-outputs/reports/long_campaign_frozen_baseline_validation.json}"

source .venv/bin/activate
PSMN_FROZEN_VALIDATION_OUTPUT="$FROZEN_VALIDATION_OUTPUT" \
PSMN_FROZEN_VALIDATION_CSV="$FROZEN_VALIDATION_CSV" \
PSMN_FROZEN_VALIDATION_JSON="$FROZEN_VALIDATION_JSON" \
  bash ./scripts/run_frozen_baseline_validation.sh

python -m psmn_rl.analysis.lss_long_campaign registration \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REGISTRATION_OUTPUT"

python -m psmn_rl.analysis.lss_long_campaign baseline-sync \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --combined-csv "$COMBINED_CSV" \
  --final-csv "$FINAL_CSV" \
  --output "$BASELINE_OUTPUT"

python -m psmn_rl.analysis.lss_long_campaign mechanism-shortlist \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$SHORTLIST_OUTPUT"
