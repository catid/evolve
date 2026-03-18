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
print(campaign["frozen_pack"])
print(reports["stage1_csv"])
print(reports["candidate_summary_markdown"])
print(reports["candidate_metrics_json"])
print(reports["combined_report_markdown"])
print(reports["combined_report_csv"])
print(reports["retry_report_markdown"])
print(reports["retry_report_csv"])
print(reports["candidate_pack_json"])
print(reports["gate_report_markdown"])
print(reports["gate_report_json"])
print(reports["stage1_json"])
print(reports["stage2_json"])
print(reports["stage3_json"])
print(reports["stage4_json"])
print(reports["decision_memo"])
PY
)

FROZEN_PACK="${paths[0]}"

python -m psmn_rl.analysis.lss_post_pass_campaign candidate-pack \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage1-csv "${paths[1]}" \
  --summary-output "${paths[2]}" \
  --metrics-output "${paths[3]}" \
  --combined-report-output "${paths[4]}" \
  --combined-report-csv "${paths[5]}" \
  --retry-report-output "${paths[6]}" \
  --retry-report-csv "${paths[7]}" \
  --candidate-pack-output "${paths[8]}"

python -m psmn_rl.analysis.claim_gate \
  --frozen-pack "$FROZEN_PACK" \
  --candidate-pack "${paths[8]}" \
  --output "${paths[9]}" \
  --json-output "${paths[10]}"

python -m psmn_rl.analysis.lss_post_pass_campaign decision-memo \
  --stage1-json "${paths[11]}" \
  --stage2-json "${paths[12]}" \
  --stage3-json "${paths[13]}" \
  --stage4-json "${paths[14]}" \
  --gate-json "${paths[10]}" \
  --output "${paths[15]}"
