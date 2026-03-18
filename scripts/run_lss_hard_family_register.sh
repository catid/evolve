#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_hard_family_program/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
reports = campaign["reports"]
print(reports["definition"])
print(reports["registration"])
print(reports["baseline_sync"])
print(reports["stage1_shortlist"])
PY
)

DEFINITION="${cfg[0]}"
REGISTRATION="${cfg[1]}"
BASELINE="${cfg[2]}"
SHORTLIST="${cfg[3]}"

python -m psmn_rl.analysis.lss_hard_family_program definition \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$DEFINITION"

python -m psmn_rl.analysis.lss_hard_family_program registration \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REGISTRATION"

python -m psmn_rl.analysis.lss_hard_family_program baseline-sync \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$BASELINE"

python -m psmn_rl.analysis.lss_hard_family_program shortlist \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$SHORTLIST"
