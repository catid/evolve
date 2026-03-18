#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_successor_challenge/campaign.yaml}"

source .venv/bin/activate
REPORT_OUTPUT=$(python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(campaign["reports"]["registration"])
PY
)

python -m psmn_rl.analysis.lss_successor_challenge registration \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT"
