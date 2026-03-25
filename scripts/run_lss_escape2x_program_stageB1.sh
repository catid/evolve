#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_escape2x_program/campaign.yaml}"

source .venv/bin/activate
./.venv/bin/python -m psmn_rl.analysis.lss_escape2x run-rescue-stage \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage screening \
  --device "$DEVICE"

readarray -t cfg < <(
  ./.venv/bin/python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["rescue_stage1_report"])
print(campaign["reports"]["rescue_stage1_csv"])
print(campaign["reports"]["rescue_stage1_json"])
PY
)

REPORT_OUTPUT="${cfg[0]}"
REPORT_CSV="${cfg[1]}"
REPORT_JSON="${cfg[2]}"

./.venv/bin/python -m psmn_rl.analysis.lss_escape2x rescue-screening-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
