#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_next_round_campaign/campaign.yaml}"
LIVE_GATE_MD="outputs/reports/portfolio_gate_report.md"
SOURCE_PACK="outputs/reports/next_mega_portfolio_candidate_pack.json"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["state_reconciliation"])
print(campaign["reports"]["registration"])
print(campaign["reports"]["baseline_sync"])
print(campaign["current_claim_manifest"])
print(campaign["current_gate_reference_pack_markdown"])
print(campaign["current_gate_reference_pack"])
print(campaign["current_gate_reference_validation"])
print(campaign["current_gate_reference_validation_json"])
print(campaign["current_claim_ledger_snapshot"])
print(campaign["current_canonical_pack"])
print(campaign["current_canonical_gate_report"])
PY
)

STATE_OUTPUT="${cfg[0]}"
REGISTRATION_OUTPUT="${cfg[1]}"
BASELINE_OUTPUT="${cfg[2]}"
CURRENT_MANIFEST="${cfg[3]}"
CURRENT_PACK_MD="${cfg[4]}"
CURRENT_PACK_JSON="${cfg[5]}"
CURRENT_PACK_VALIDATION_MD="${cfg[6]}"
CURRENT_PACK_VALIDATION_JSON="${cfg[7]}"
LEDGER_SNAPSHOT="${cfg[8]}"
LIVE_PACK_JSON="${cfg[9]}"
LIVE_GATE_JSON="${cfg[10]}"

python -m psmn_rl.analysis.round6_state_repair snapshot-claim-ledger --output "$LEDGER_SNAPSHOT"

python -m psmn_rl.analysis.benchmark_pack seal-frozen-pack \
  --manifest "$CURRENT_MANIFEST" \
  --output-markdown "$CURRENT_PACK_MD" \
  --output-json "$CURRENT_PACK_JSON"

python -m psmn_rl.analysis.round6_state_repair refresh-live-pack \
  --source-pack "$SOURCE_PACK" \
  --gate-reference-pack "$CURRENT_PACK_JSON" \
  --output "$LIVE_PACK_JSON"

python -m psmn_rl.analysis.round6_state_repair state-reconciliation --output "$STATE_OUTPUT"
python -m psmn_rl.analysis.round6_state_repair baseline-sync --output "$BASELINE_OUTPUT"

python -m psmn_rl.analysis.benchmark_pack seal-frozen-pack \
  --manifest "$CURRENT_MANIFEST" \
  --output-markdown "$CURRENT_PACK_MD" \
  --output-json "$CURRENT_PACK_JSON"

python -m psmn_rl.analysis.benchmark_pack validate-frozen-pack \
  --pack "$CURRENT_PACK_JSON" \
  --output "$CURRENT_PACK_VALIDATION_MD" \
  --json-output "$CURRENT_PACK_VALIDATION_JSON"

python -m psmn_rl.analysis.round6_state_repair refresh-live-pack \
  --source-pack "$SOURCE_PACK" \
  --gate-reference-pack "$CURRENT_PACK_JSON" \
  --output "$LIVE_PACK_JSON"

python -m psmn_rl.analysis.claim_gate \
  --frozen-pack "$CURRENT_PACK_JSON" \
  --candidate-pack "$LIVE_PACK_JSON" \
  --output "$LIVE_GATE_MD" \
  --json-output "$LIVE_GATE_JSON"

python -m psmn_rl.analysis.round6_state_repair gate-repair \
  --validation-json "$CURRENT_PACK_VALIDATION_JSON" \
  --gate-json "$LIVE_GATE_JSON" \
  --output outputs/reports/next_round_gate_repair.md

python -m psmn_rl.analysis.lss_portfolio_campaign registration \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REGISTRATION_OUTPUT"
