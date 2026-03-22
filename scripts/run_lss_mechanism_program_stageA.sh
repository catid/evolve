#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_mechanism_program/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["stageA1_report"])
print(campaign["reports"]["stageA1_json"])
print(campaign["reports"]["stageA2_report"])
print(campaign["reports"]["stageA2_csv"])
print(campaign["reports"]["stageA2_json"])
print(campaign["reports"]["stageA3_report"])
print(campaign["reports"]["stageA3_json"])
print(campaign["reports"]["stageA4_report"])
print(campaign["reports"]["stageA4_json"])
print(campaign["reports"]["stageA5_report"])
print(campaign["reports"]["stageA5_json"])
print(campaign["analysis"]["phase_trace_episodes"])
print(campaign["analysis"]["phase_max_steps"])
PY
)

A1_OUTPUT="${cfg[0]}"
A1_JSON="${cfg[1]}"
A2_OUTPUT="${cfg[2]}"
A2_CSV="${cfg[3]}"
A2_JSON="${cfg[4]}"
A3_OUTPUT="${cfg[5]}"
A3_JSON="${cfg[6]}"
A4_OUTPUT="${cfg[7]}"
A4_JSON="${cfg[8]}"
A5_OUTPUT="${cfg[9]}"
A5_JSON="${cfg[10]}"
TRACE_EPISODES="${cfg[11]}"
TRACE_MAX_STEPS="${cfg[12]}"

python -m psmn_rl.analysis.lss_mechanism_program round-differential \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$A1_OUTPUT" \
  --json "$A1_JSON"

python -m psmn_rl.analysis.lss_mechanism_program phase-local \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$A2_OUTPUT" \
  --csv "$A2_CSV" \
  --json "$A2_JSON" \
  --episodes "$TRACE_EPISODES" \
  --max-steps "$TRACE_MAX_STEPS"

python -m psmn_rl.analysis.lss_mechanism_program selection-sensitivity \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$A3_OUTPUT" \
  --json "$A3_JSON"

python -m psmn_rl.analysis.lss_mechanism_program restart-sensitivity \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$A4_OUTPUT" \
  --json "$A4_JSON"

python -m psmn_rl.analysis.lss_mechanism_program shortlist \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$A5_OUTPUT" \
  --json "$A5_JSON"
