#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_post_pass_campaign/campaign.yaml}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"

source .venv/bin/activate
readarray -t paths < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
reports = campaign["reports"]
print(reports["stage1_json"])
print(reports["stage4_report"])
print(reports["stage4_csv"])
print(reports["stage4_json"])
PY
)

python -m psmn_rl.analysis.lss_post_pass_campaign stage4-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage1-json "${paths[0]}" \
  --device "$DEVICE" \
  --episodes "$EPISODES" \
  --output "${paths[1]}" \
  --csv "${paths[2]}" \
  --json "${paths[3]}"
