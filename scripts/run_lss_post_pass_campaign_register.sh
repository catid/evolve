#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_post_pass_campaign/campaign.yaml}"
source .venv/bin/activate

readarray -t campaign_values < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
reports = campaign["reports"]
print(reports["registration"])
print(reports["baseline_sync"])
PY
)

REGISTRATION_OUTPUT="${campaign_values[0]}"
BASELINE_SYNC_OUTPUT="${campaign_values[1]}"

python -m psmn_rl.analysis.lss_post_pass_campaign registration \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REGISTRATION_OUTPUT"

python -m psmn_rl.analysis.lss_post_pass_campaign baseline-sync \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$BASELINE_SYNC_OUTPUT"
