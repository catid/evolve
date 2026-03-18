#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_canonization_campaign/campaign.yaml}"

source .venv/bin/activate
readarray -t paths < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
reports = campaign["reports"]
print(reports["registration"])
print(reports["baseline_sync"])
print(reports["stage1_shortlist"])
PY
)

python -m psmn_rl.analysis.lss_canonization_campaign registration \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "${paths[0]}"

python -m psmn_rl.analysis.lss_canonization_campaign baseline-sync \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "${paths[1]}"

python -m psmn_rl.analysis.lss_canonization_campaign mechanism-shortlist \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "${paths[2]}"
