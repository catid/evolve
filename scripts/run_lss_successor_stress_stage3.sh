#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_successor_stress/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
stage1 = json.loads(Path(campaign["reports"]["stage1_json"]).read_text())
stage2 = json.loads(Path(campaign["reports"]["stage2_json"]).read_text())
selected = stage2["selected_candidate"]
route_case = stage1["selected_route_case"]
run_dir = Path(campaign["stage_roots"]["stage1_screening"]) / selected / route_case["lane"] / f"seed_{route_case['seed']}" / "kl_lss_sare"
print(route_case["lane"])
print(route_case["seed"])
print(run_dir)
print(campaign["reports"]["route_raw_markdown"])
print(campaign["reports"]["route_raw_csv"])
print(campaign["reports"]["stage3_report"])
print(campaign["reports"]["stage3_csv"])
print(campaign["reports"]["stage3_json"])
print(campaign["reports"]["decision_memo"])
print(campaign["reports"]["stage1_json"])
print(campaign["reports"]["stage2_json"])
PY
)

LANE="${cfg[0]}"
SEED="${cfg[1]}"
RUN_DIR="${cfg[2]}"
ROUTE_RAW_MD="${cfg[3]}"
ROUTE_RAW_CSV="${cfg[4]}"
STAGE3_REPORT="${cfg[5]}"
STAGE3_REPORT_CSV="${cfg[6]}"
STAGE3_REPORT_JSON="${cfg[7]}"
DECISION_MEMO="${cfg[8]}"
STAGE1_JSON="${cfg[9]}"
STAGE2_JSON="${cfg[10]}"

python -m psmn_rl.analysis.lss_route_dependence \
  --case "$LANE" "$SEED" "$RUN_DIR" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$ROUTE_RAW_MD" \
  --csv "$ROUTE_RAW_CSV"

python -m psmn_rl.analysis.lss_successor_stress stage3-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage1-json "$STAGE1_JSON" \
  --stage2-json "$STAGE2_JSON" \
  --route-csv "$ROUTE_RAW_CSV" \
  --output "$STAGE3_REPORT" \
  --csv "$STAGE3_REPORT_CSV" \
  --json "$STAGE3_REPORT_JSON"

python -m psmn_rl.analysis.lss_successor_stress decision-memo \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage1-json "$STAGE1_JSON" \
  --stage2-json "$STAGE2_JSON" \
  --stage3-json "$STAGE3_REPORT_JSON" \
  --output "$DECISION_MEMO"
