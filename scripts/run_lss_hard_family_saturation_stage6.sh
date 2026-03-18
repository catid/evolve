#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_hard_family_saturation/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
reports = campaign["reports"]
print(reports["stage3_json"])
print(reports["stage4_json"])
print(reports["stage5_json"])
print(reports["stage6_json"])
print(reports["stage7_report"])
print(reports["stage7_csv"])
print(reports["stage7_json"])
PY
)

python -m psmn_rl.analysis.lss_hard_family_saturation stage7-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage3-json "${cfg[0]}" \
  --stage4-json "${cfg[1]}" \
  --stage5-json "${cfg[2]}" \
  --stage6-json "${cfg[3]}" \
  --device "$DEVICE" \
  --episodes "$EPISODES" \
  --output "${cfg[4]}" \
  --csv "${cfg[5]}" \
  --json "${cfg[6]}"
