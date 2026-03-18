#!/usr/bin/env bash
set -euo pipefail

MANIFEST_PATH="${PSMN_MANIFEST_PATH:-configs/claims/doorkey_frozen_claim.yaml}"
FROZEN_PACK="${PSMN_FROZEN_PACK:-outputs/reports/frozen_benchmark_pack.json}"
BASELINE_COMBINED_CSV="${PSMN_BASELINE_COMBINED_CSV:-outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.csv}"
STAGE3_CSV="${PSMN_STAGE3_CSV:-outputs/reports/long_campaign_stage3_fairness.csv}"
STAGE3_JSON="${PSMN_STAGE3_JSON:-outputs/reports/long_campaign_stage3_fairness.json}"
STAGE4_CSV="${PSMN_STAGE4_CSV:-outputs/reports/long_campaign_stage4_replication.csv}"
CANDIDATE_SUMMARY="${PSMN_CANDIDATE_SUMMARY:-outputs/reports/long_campaign_candidate_summary.md}"
CANDIDATE_METRICS="${PSMN_CANDIDATE_METRICS:-outputs/reports/long_campaign_candidate_metrics.json}"
COMBINED_REPORT="${PSMN_COMBINED_REPORT:-outputs/reports/long_campaign_candidate_combined_report.md}"
COMBINED_CSV="${PSMN_COMBINED_CSV:-outputs/reports/long_campaign_candidate_combined_report.csv}"
RETRY_REPORT="${PSMN_RETRY_REPORT:-outputs/reports/long_campaign_candidate_retry_block_report.md}"
RETRY_CSV="${PSMN_RETRY_CSV:-outputs/reports/long_campaign_candidate_retry_block_report.csv}"
CANDIDATE_PACK="${PSMN_CANDIDATE_PACK:-outputs/reports/long_campaign_candidate_pack.json}"
GATE_REPORT="${PSMN_GATE_REPORT:-outputs/reports/long_campaign_gate_report.md}"
GATE_JSON="${PSMN_GATE_JSON:-outputs/reports/long_campaign_gate_report.json}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_long_campaign candidate-pack \
  --manifest "$MANIFEST_PATH" \
  --frozen-pack "$FROZEN_PACK" \
  --baseline-combined-csv "$BASELINE_COMBINED_CSV" \
  --stage3-csv "$STAGE3_CSV" \
  --stage3-json "$STAGE3_JSON" \
  --stage4-csv "$STAGE4_CSV" \
  --candidate-summary-output "$CANDIDATE_SUMMARY" \
  --candidate-metrics-output "$CANDIDATE_METRICS" \
  --combined-report-output "$COMBINED_REPORT" \
  --combined-report-csv "$COMBINED_CSV" \
  --retry-report-output "$RETRY_REPORT" \
  --retry-report-csv "$RETRY_CSV" \
  --candidate-pack-output "$CANDIDATE_PACK"

python -m psmn_rl.analysis.claim_gate \
  --frozen-pack "$FROZEN_PACK" \
  --candidate-pack "$CANDIDATE_PACK" \
  --output "$GATE_REPORT" \
  --json-output "$GATE_JSON"
