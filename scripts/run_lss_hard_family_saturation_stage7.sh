#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_hard_family_saturation/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
reports = campaign["reports"]
print(campaign["frozen_pack"])
print(reports["stage2_json"])
print(reports["stage3_json"])
print(reports["stage4_json"])
print(reports["stage5_json"])
print(reports["stage6_json"])
print(reports["stage7_json"])
print(reports["candidate_summary_markdown"])
print(reports["candidate_metrics_json"])
print(reports["combined_report_markdown"])
print(reports["combined_report_csv"])
print(reports["retry_report_markdown"])
print(reports["retry_report_csv"])
print(reports["successor_pack_markdown"])
print(reports["successor_pack_json"])
print(reports["gate_report_markdown"])
print(reports["gate_report_json"])
print(reports["decision_memo"])
PY
)

FROZEN_PACK="${cfg[0]}"
STAGE1_JSON="${cfg[1]}"
STAGE2_JSON="${cfg[2]}"
STAGE3_JSON="${cfg[3]}"
STAGE4_JSON="${cfg[4]}"
STAGE5_JSON="${cfg[5]}"
STAGE6_JSON="${cfg[6]}"
SUMMARY_OUTPUT="${cfg[7]}"
METRICS_OUTPUT="${cfg[8]}"
COMBINED_OUTPUT="${cfg[9]}"
COMBINED_CSV="${cfg[10]}"
RETRY_OUTPUT="${cfg[11]}"
RETRY_CSV="${cfg[12]}"
SUCCESSOR_MD="${cfg[13]}"
SUCCESSOR_JSON="${cfg[14]}"
GATE_MD="${cfg[15]}"
GATE_JSON="${cfg[16]}"
DECISION_MEMO="${cfg[17]}"

python -m psmn_rl.analysis.lss_hard_family_saturation successor-pack \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage3-json "$STAGE2_JSON" \
  --stage4-json "$STAGE3_JSON" \
  --stage5-json "$STAGE4_JSON" \
  --stage6-json "$STAGE5_JSON" \
  --stage7-json "$STAGE6_JSON" \
  --summary-output "$SUMMARY_OUTPUT" \
  --metrics-output "$METRICS_OUTPUT" \
  --combined-report-output "$COMBINED_OUTPUT" \
  --combined-report-csv "$COMBINED_CSV" \
  --retry-report-output "$RETRY_OUTPUT" \
  --retry-report-csv "$RETRY_CSV" \
  --successor-pack-markdown "$SUCCESSOR_MD" \
  --successor-pack-json "$SUCCESSOR_JSON"

PACK_STATUS="$(
  source .venv/bin/activate
  python - "$SUCCESSOR_JSON" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
print(payload.get("status", "ok"))
PY
)"

if [[ "$PACK_STATUS" == "ok" ]]; then
  python -m psmn_rl.analysis.claim_gate \
    --frozen-pack "$FROZEN_PACK" \
    --candidate-pack "$SUCCESSOR_JSON" \
    --output "$GATE_MD" \
    --json-output "$GATE_JSON"
fi

if [[ -f "$GATE_JSON" ]]; then
  python -m psmn_rl.analysis.lss_hard_family_saturation decision-memo \
    --stage2-json "$STAGE1_JSON" \
    --stage3-json "$STAGE2_JSON" \
    --stage4-json "$STAGE3_JSON" \
    --stage5-json "$STAGE4_JSON" \
    --stage6-json "$STAGE5_JSON" \
    --stage7-json "$STAGE6_JSON" \
    --gate-json "$GATE_JSON" \
    --output "$DECISION_MEMO"
else
  python -m psmn_rl.analysis.lss_hard_family_saturation decision-memo \
    --stage2-json "$STAGE1_JSON" \
    --stage3-json "$STAGE2_JSON" \
    --stage4-json "$STAGE3_JSON" \
    --stage5-json "$STAGE4_JSON" \
    --stage6-json "$STAGE5_JSON" \
    --stage7-json "$STAGE6_JSON" \
    --output "$DECISION_MEMO"
fi
