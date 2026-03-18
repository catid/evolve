#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_post_pass_campaign/campaign.yaml}"

source .venv/bin/activate
readarray -t paths < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
reports = campaign["reports"]
print(campaign["historical_candidate_roots"]["strong"])
print(campaign["historical_candidate_roots"]["weak"])
print(reports["stage3_raw"])
print(reports["stage3_raw_csv"])
print(reports["stage3_report"])
print(reports["stage3_csv"])
print(reports["stage3_json"])
PY
)

STRONG_ROOT="${paths[0]}"
WEAK_ROOT="${paths[1]}"
ROUTE_RAW="${paths[2]}"
ROUTE_CSV="${paths[3]}"
REPORT_OUTPUT="${paths[4]}"
REPORT_CSV="${paths[5]}"
REPORT_JSON="${paths[6]}"

python -m psmn_rl.analysis.lss_route_dependence \
  --case original 7 "${STRONG_ROOT}/original/seed_7/kl_lss_sare" \
  --case fresh 29 "${STRONG_ROOT}/fresh/seed_29/kl_lss_sare" \
  --case fresh_final 47 "${WEAK_ROOT}/fresh_final/seed_47/kl_lss_sare" \
  --case fresh_final 59 "${WEAK_ROOT}/fresh_final/seed_59/kl_lss_sare" \
  --episodes 64 \
  --device "$DEVICE" \
  --output "$ROUTE_RAW" \
  --csv "$ROUTE_CSV"

python -m psmn_rl.analysis.lss_post_pass_campaign stage3-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --route-csv "$ROUTE_CSV" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
