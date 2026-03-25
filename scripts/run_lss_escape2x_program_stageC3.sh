#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_escape2x_program/campaign.yaml}"

source .venv/bin/activate
if ./.venv/bin/python - "$CAMPAIGN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
payload = json.loads(Path(campaign["reports"]["practicalization_stage2_json"]).read_text(encoding="utf-8"))
raise SystemExit(0 if bool(payload.get("holdout_survivors")) else 1)
PY
then
  PSMN_CAMPAIGN_CONFIG="$CAMPAIGN_CONFIG" bash ./scripts/run_lss_expansion_mega_program_stage6.sh
  PSMN_CAMPAIGN_CONFIG="$CAMPAIGN_CONFIG" bash ./scripts/run_lss_expansion_mega_program_stage7.sh
fi

readarray -t cfg < <(
  ./.venv/bin/python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["practicalization_stage3_report"])
print(campaign["reports"]["practicalization_stage3_json"])
PY
)

REPORT_OUTPUT="${cfg[0]}"
REPORT_JSON="${cfg[1]}"

./.venv/bin/python -m psmn_rl.analysis.lss_escape2x practicalization-route-stability-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --json "$REPORT_JSON"
