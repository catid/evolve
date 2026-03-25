#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_escape_distill_program/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
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

python -m psmn_rl.analysis.lss_escape_distill candidate-pack \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$PACK_OUTPUT"

python -m psmn_rl.analysis.claim_gate \
  --frozen-pack "$FROZEN_PACK" \
  --candidate-pack "$PACK_OUTPUT" \
  --output "$GATE_MD" \
  --json-output "$GATE_JSON"

python -m psmn_rl.analysis.lss_escape_distill decision-memo \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$DECISION_MEMO"
