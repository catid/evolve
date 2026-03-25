#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_escape_distill_program/campaign.yaml}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_escape_distill run-rescue-stage \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage rerun \
  --device "$DEVICE"

readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["rescue_stage2_report"])
print(campaign["reports"]["rescue_stage2_csv"])
print(campaign["reports"]["rescue_stage2_json"])
PY
)

REPORT_OUTPUT="${cfg[0]}"
REPORT_CSV="${cfg[1]}"
REPORT_JSON="${cfg[2]}"

python -m psmn_rl.analysis.lss_escape_distill rescue-rerun-holdout-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
