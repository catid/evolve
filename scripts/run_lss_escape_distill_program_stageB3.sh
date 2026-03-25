#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_escape_distill_program/campaign.yaml}"

source .venv/bin/activate

if python - "$CAMPAIGN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
payload = json.loads(Path(campaign["reports"]["rescue_stage2_json"]).read_text(encoding="utf-8"))
raise SystemExit(0 if bool(payload.get("surviving_candidates")) else 1)
PY
then
  python -m psmn_rl.analysis.lss_escape_distill run-rescue-stage \
    --campaign-config "$CAMPAIGN_CONFIG" \
    --stage antiregression \
    --device "$DEVICE"

  readarray -t route_cfg < <(
    python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config
from psmn_rl.analysis.lss_deadlock_program import _round6_run_dir

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["rescue_route_raw_report"])
print(campaign["reports"]["rescue_route_raw_csv"])
print(_round6_run_dir(campaign, "prospective_c", 191))
print(_round6_run_dir(campaign, "prospective_g", 257))
print(_round6_run_dir(campaign, "fresh", 23))
PY
  )

  ROUTE_MD="${route_cfg[0]}"
  ROUTE_CSV="${route_cfg[1]}"
  DEV_RUN="${route_cfg[2]}"
  HOLDOUT_RUN="${route_cfg[3]}"
  HEALTHY_RUN="${route_cfg[4]}"

  python -m psmn_rl.analysis.lss_route_dependence \
    --case prospective_c 191 "$DEV_RUN" \
    --case prospective_g 257 "$HOLDOUT_RUN" \
    --case fresh 23 "$HEALTHY_RUN" \
    --episodes 32 \
    --device "$DEVICE" \
    --output "$ROUTE_MD" \
    --csv "$ROUTE_CSV"
fi

readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
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

python -m psmn_rl.analysis.lss_escape_distill rescue-antiregression-route-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
