#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_escape_distill_program/campaign.yaml}"

source .venv/bin/activate
./scripts/run_lss_escape_distill_program_register.sh

readarray -t cfg < <(
  ./.venv/bin/python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["subgroup_detector_report"])
print(campaign["reports"]["subgroup_detector_json"])
print(campaign["reports"]["source_quality_report"])
print(campaign["reports"]["source_quality_json"])
print(campaign["reports"]["registration_report"])
print(campaign["reports"]["registration_json"])
PY
)

DETECTOR_OUTPUT="${cfg[0]}"
DETECTOR_JSON="${cfg[1]}"
SOURCE_QUALITY_OUTPUT="${cfg[2]}"
SOURCE_QUALITY_JSON="${cfg[3]}"
REGISTRATION_OUTPUT="${cfg[4]}"
REGISTRATION_JSON="${cfg[5]}"

./.venv/bin/python -m psmn_rl.analysis.lss_escape_distill subgroup-detector \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$DETECTOR_OUTPUT" \
  --json "$DETECTOR_JSON" \
  --device "$DEVICE"

./.venv/bin/python -m psmn_rl.analysis.lss_escape_distill source-quality-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$SOURCE_QUALITY_OUTPUT" \
  --json "$SOURCE_QUALITY_JSON" \
  --device "$DEVICE"

./.venv/bin/python -m psmn_rl.analysis.lss_escape_distill registration \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REGISTRATION_OUTPUT" \
  --json "$REGISTRATION_JSON"
