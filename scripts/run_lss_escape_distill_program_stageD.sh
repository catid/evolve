#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_escape_distill_program/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["archpilot_report"])
print(campaign["reports"]["archpilot_json"])
PY
)

REPORT_OUTPUT="${cfg[0]}"
REPORT_JSON="${cfg[1]}"

python -m psmn_rl.analysis.lss_escape_distill archpilot-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --json "$REPORT_JSON"
