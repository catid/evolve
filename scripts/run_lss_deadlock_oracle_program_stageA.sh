#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_deadlock_oracle_program/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["family_definition"])
print(campaign["reports"]["teacher_audit_report"])
print(campaign["reports"]["teacher_audit_json"])
print(campaign["reports"]["distribution_audit_report"])
print(campaign["reports"]["distribution_audit_json"])
print(campaign["reports"]["oracle_teacher_target_report"])
print(campaign["reports"]["oracle_teacher_target_json"])
print(campaign["reports"]["oracle_transition_report"])
print(campaign["reports"]["oracle_transition_json"])
print(campaign["reports"]["oracle_combined_report"])
print(campaign["reports"]["oracle_combined_json"])
print(campaign["reports"]["oracle_inference_report"])
print(campaign["reports"]["oracle_inference_csv"])
print(campaign["reports"]["oracle_inference_json"])
print(campaign["reports"]["oracle_synthesis_report"])
print(campaign["reports"]["oracle_synthesis_json"])
print(campaign["reports"]["oracle_shortlist_report"])
print(campaign["reports"]["oracle_shortlist_json"])
print(campaign["analysis"]["casebook_trace_episodes"])
print(campaign["analysis"]["casebook_max_steps"])
PY
)

FAMILY_OUTPUT="${cfg[0]}"
TEACHER_OUTPUT="${cfg[1]}"
TEACHER_JSON="${cfg[2]}"
DISTRIBUTION_OUTPUT="${cfg[3]}"
DISTRIBUTION_JSON="${cfg[4]}"
ORACLE_A2_OUTPUT="${cfg[5]}"
ORACLE_A2_JSON="${cfg[6]}"
ORACLE_A3_OUTPUT="${cfg[7]}"
ORACLE_A3_JSON="${cfg[8]}"
ORACLE_A4_OUTPUT="${cfg[9]}"
ORACLE_A4_JSON="${cfg[10]}"
ORACLE_A5_OUTPUT="${cfg[11]}"
ORACLE_A5_CSV="${cfg[12]}"
ORACLE_A5_JSON="${cfg[13]}"
ORACLE_A6_OUTPUT="${cfg[14]}"
ORACLE_A6_JSON="${cfg[15]}"
SHORTLIST_OUTPUT="${cfg[16]}"
SHORTLIST_JSON="${cfg[17]}"
TRACE_EPISODES="${cfg[18]}"
TRACE_MAX_STEPS="${cfg[19]}"

python -m psmn_rl.analysis.lss_deadlock_program family-definition \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$FAMILY_OUTPUT"

python -m psmn_rl.analysis.lss_deadlock_program teacher-audit \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$TEACHER_OUTPUT" \
  --json "$TEACHER_JSON" \
  --episodes "$TRACE_EPISODES" \
  --max-steps "$TRACE_MAX_STEPS" \
  --device "$DEVICE"

python -m psmn_rl.analysis.lss_deadlock_program distribution-audit \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$DISTRIBUTION_OUTPUT" \
  --json "$DISTRIBUTION_JSON"

python -m psmn_rl.analysis.lss_deadlock_oracle run-oracles \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --device "$DEVICE"

python -m psmn_rl.analysis.lss_deadlock_oracle teacher-target-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$ORACLE_A2_OUTPUT" \
  --json "$ORACLE_A2_JSON"

python -m psmn_rl.analysis.lss_deadlock_oracle transition-coverage-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$ORACLE_A3_OUTPUT" \
  --json "$ORACLE_A3_JSON"

python -m psmn_rl.analysis.lss_deadlock_oracle combined-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$ORACLE_A4_OUTPUT" \
  --json "$ORACLE_A4_JSON"

python -m psmn_rl.analysis.lss_deadlock_oracle inference-escape-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$ORACLE_A5_OUTPUT" \
  --csv "$ORACLE_A5_CSV" \
  --json "$ORACLE_A5_JSON" \
  --device "$DEVICE"

python -m psmn_rl.analysis.lss_deadlock_oracle synthesis-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$ORACLE_A6_OUTPUT" \
  --shortlist-output "$SHORTLIST_OUTPUT" \
  --json "$ORACLE_A6_JSON" \
  --shortlist-json "$SHORTLIST_JSON"
