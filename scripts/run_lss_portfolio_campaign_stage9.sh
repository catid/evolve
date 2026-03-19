#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_portfolio_campaign/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(campaign["reports"]["candidate_pack_json"])
print(campaign["reports"]["gate_report_markdown"])
print(campaign["reports"]["gate_report_json"])
print(campaign["reports"]["decision_memo"])
print(campaign["frozen_pack"])
PY
)

PACK_OUTPUT="${cfg[0]}"
GATE_MD="${cfg[1]}"
GATE_JSON="${cfg[2]}"
DECISION_MEMO="${cfg[3]}"
FROZEN_PACK="${cfg[4]}"

PSMN_CAMPAIGN_CONFIG="$CAMPAIGN_CONFIG" bash ./scripts/run_lss_successor_migration_stage7.sh

python -m psmn_rl.analysis.lss_portfolio_campaign refresh-pack \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$PACK_OUTPUT"

python -m psmn_rl.analysis.claim_gate \
  --frozen-pack "$FROZEN_PACK" \
  --candidate-pack "$PACK_OUTPUT" \
  --output "$GATE_MD" \
  --json-output "$GATE_JSON"

python -m psmn_rl.analysis.lss_portfolio_campaign decision-memo \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$DECISION_MEMO"
