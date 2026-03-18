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
print(reports["stage2_json"])
print(reports["stage3_json"])
print(reports["stage4_json"])
print(reports["successor_pack_markdown"])
print(reports["successor_pack_json"])
PY
)

python -m psmn_rl.analysis.lss_post_pass_campaign successor-draft \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage2-json "${paths[0]}" \
  --stage3-json "${paths[1]}" \
  --stage4-json "${paths[2]}" \
  --output "${paths[3]}" \
  --json "${paths[4]}"
