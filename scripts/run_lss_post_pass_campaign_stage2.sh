#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_post_pass_campaign/campaign.yaml}"

source .venv/bin/activate
readarray -t paths < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
reports = campaign["reports"]
print(reports["stage1_csv"])
print(reports["stage2_report"])
print(reports["stage2_csv"])
print(reports["stage2_json"])
PY
)

python -m psmn_rl.analysis.lss_post_pass_campaign stage2-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage1-csv "${paths[0]}" \
  --output "${paths[1]}" \
  --csv "${paths[2]}" \
  --json "${paths[3]}"
