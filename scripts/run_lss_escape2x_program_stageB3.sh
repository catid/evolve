#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_escape2x_program/campaign.yaml}"

source .venv/bin/activate

if ./.venv/bin/python - "$CAMPAIGN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
payload = json.loads(Path(campaign["reports"]["rescue_stage2_json"]).read_text(encoding="utf-8"))
raise SystemExit(0 if bool(payload.get("surviving_candidates")) else 1)
PY
then
  ./.venv/bin/python -m psmn_rl.analysis.lss_escape2x run-rescue-stage \
    --campaign-config "$CAMPAIGN_CONFIG" \
    --stage antiregression \
    --device "$DEVICE"

  readarray -t route_cfg < <(
    ./.venv/bin/python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config
from psmn_rl.analysis.lss_deadlock_program import _round6_run_dir

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["rescue_route_raw_report"])
print(campaign["reports"]["rescue_route_raw_csv"])
dev_lane, dev_seed = campaign["analysis"]["teacher_locked_dev_groups"][0]["cases"][0]
holdout_lane, holdout_seed = campaign["analysis"]["teacher_locked_holdout_groups"][0]["cases"][0]
healthy_lane, healthy_seed = campaign["analysis"]["healthy_groups"][0]["cases"][0]
print(dev_lane)
print(dev_seed)
print(_round6_run_dir(campaign, dev_lane, int(dev_seed)))
print(holdout_lane)
print(holdout_seed)
print(_round6_run_dir(campaign, holdout_lane, int(holdout_seed)))
print(healthy_lane)
print(healthy_seed)
print(_round6_run_dir(campaign, healthy_lane, int(healthy_seed)))
PY
  )

  ROUTE_MD="${route_cfg[0]}"
  ROUTE_CSV="${route_cfg[1]}"
  DEV_LANE="${route_cfg[2]}"
  DEV_SEED="${route_cfg[3]}"
  DEV_RUN="${route_cfg[4]}"
  HOLDOUT_LANE="${route_cfg[5]}"
  HOLDOUT_SEED="${route_cfg[6]}"
  HOLDOUT_RUN="${route_cfg[7]}"
  HEALTHY_LANE="${route_cfg[8]}"
  HEALTHY_SEED="${route_cfg[9]}"
  HEALTHY_RUN="${route_cfg[10]}"

  ./.venv/bin/python -m psmn_rl.analysis.lss_route_dependence \
    --case "$DEV_LANE" "$DEV_SEED" "$DEV_RUN" \
    --case "$HOLDOUT_LANE" "$HOLDOUT_SEED" "$HOLDOUT_RUN" \
    --case "$HEALTHY_LANE" "$HEALTHY_SEED" "$HEALTHY_RUN" \
    --episodes 32 \
    --device "$DEVICE" \
    --output "$ROUTE_MD" \
    --csv "$ROUTE_CSV"
fi

readarray -t cfg < <(
  ./.venv/bin/python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["rescue_stage3_report"])
print(campaign["reports"]["rescue_stage3_csv"])
print(campaign["reports"]["rescue_stage3_json"])
PY
)

REPORT_OUTPUT="${cfg[0]}"
REPORT_CSV="${cfg[1]}"
REPORT_JSON="${cfg[2]}"

./.venv/bin/python -m psmn_rl.analysis.lss_escape2x rescue-antiregression-route-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
