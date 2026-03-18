#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_successor_challenge/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
stage2 = json.loads(Path(campaign["reports"]["stage2_json"]).read_text())
selected = stage2["selected_challenger"]
stage1_root = Path(campaign["stage_roots"]["stage1_screening"])
hard_case = campaign["route_cases"]["hard"]
healthy_case = campaign["route_cases"]["healthy"]
print(selected)
print(stage1_root / selected / hard_case["lane"] / f"seed_{hard_case['seed']}" / "kl_lss_sare")
print(stage1_root / selected / healthy_case["lane"] / f"seed_{healthy_case['seed']}" / "kl_lss_sare")
print(campaign["reports"]["route_raw_hard_markdown"])
print(campaign["reports"]["route_raw_hard_csv"])
print(campaign["reports"]["route_raw_healthy_markdown"])
print(campaign["reports"]["route_raw_healthy_csv"])
print(campaign["reports"]["stage3_report"])
print(campaign["reports"]["stage3_csv"])
print(campaign["reports"]["stage3_json"])
print(campaign["reports"]["decision_memo"])
print(hard_case["lane"])
print(hard_case["seed"])
print(healthy_case["lane"])
print(healthy_case["seed"])
PY
)

SELECTED="${cfg[0]}"
HARD_RUN_DIR="${cfg[1]}"
HEALTHY_RUN_DIR="${cfg[2]}"
HARD_RAW_MD="${cfg[3]}"
HARD_RAW_CSV="${cfg[4]}"
HEALTHY_RAW_MD="${cfg[5]}"
HEALTHY_RAW_CSV="${cfg[6]}"
REPORT_OUTPUT="${cfg[7]}"
REPORT_CSV="${cfg[8]}"
REPORT_JSON="${cfg[9]}"
DECISION_MEMO="${cfg[10]}"
HARD_LANE="${cfg[11]}"
HARD_SEED="${cfg[12]}"
HEALTHY_LANE="${cfg[13]}"
HEALTHY_SEED="${cfg[14]}"

python -m psmn_rl.analysis.lss_route_dependence \
  --case "$HARD_LANE" "$HARD_SEED" "$HARD_RUN_DIR" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$HARD_RAW_MD" \
  --csv "$HARD_RAW_CSV"

python -m psmn_rl.analysis.lss_route_dependence \
  --case "$HEALTHY_LANE" "$HEALTHY_SEED" "$HEALTHY_RUN_DIR" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$HEALTHY_RAW_MD" \
  --csv "$HEALTHY_RAW_CSV"

python -m psmn_rl.analysis.lss_successor_challenge stage3-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --hard-route-csv "$HARD_RAW_CSV" \
  --healthy-route-csv "$HEALTHY_RAW_CSV" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"

python -m psmn_rl.analysis.lss_successor_challenge decision-memo \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$DECISION_MEMO"
