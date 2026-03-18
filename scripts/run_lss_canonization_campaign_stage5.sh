#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_canonization_campaign/campaign.yaml}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
reports = campaign["reports"]
print(reports["stage3_json"])
print(reports["stage4_json"])
print(reports["stage5_route_raw"])
print(reports["stage5_route_raw_csv"])
print(reports["stage5_report"])
print(reports["stage5_csv"])
print(reports["stage5_json"])
PY
)

STAGE3_JSON="${cfg[0]}"
STAGE4_JSON="${cfg[1]}"
ROUTE_RAW="${cfg[2]}"
ROUTE_CSV="${cfg[3]}"
REPORT_OUTPUT="${cfg[4]}"
REPORT_CSV="${cfg[5]}"
REPORT_JSON="${cfg[6]}"

readarray -t cases < <(
  source .venv/bin/activate
  python - "$CAMPAIGN_CONFIG" "$STAGE3_JSON" "$STAGE4_JSON" <<'PY'
import json
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
stage3 = json.loads(Path(sys.argv[2]).read_text())
stage4 = json.loads(Path(sys.argv[3]).read_text())
best = stage3.get("best_candidate")
hard_case = stage3.get("selected_hard_case")
strong_case = stage4.get("selected_strong_case")
if not best or not hard_case or not strong_case:
    raise SystemExit(0)

def run_dir(lane, seed):
    if lane in {"post_pass_b", "post_pass_c"}:
        return Path(campaign["stage_roots"]["stage3_fairness"]) / best / lane / f"seed_{seed}" / "kl_lss_sare"
    return Path(campaign["stage_roots"]["stage4_replication"]) / best / lane / f"seed_{seed}" / "kl_lss_sare"

print("\t".join([hard_case["lane"], str(hard_case["seed"]), str(run_dir(hard_case["lane"], int(hard_case["seed"])))]))
print("\t".join([strong_case["lane"], str(strong_case["seed"]), str(run_dir(strong_case["lane"], int(strong_case["seed"])))]))
PY
)

if [[ "${#cases[@]}" -gt 0 ]]; then
  declare -a route_args=()
  for row in "${cases[@]}"; do
    IFS=$'\t' read -r lane seed run_dir <<<"$row"
    route_args+=(--case "$lane" "$seed" "$run_dir")
  done

  python -m psmn_rl.analysis.lss_route_dependence \
    "${route_args[@]}" \
    --episodes "$EPISODES" \
    --device "$DEVICE" \
    --output "$ROUTE_RAW" \
    --csv "$ROUTE_CSV"
fi

python -m psmn_rl.analysis.lss_canonization_campaign stage5-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage3-json "$STAGE3_JSON" \
  --stage4-json "$STAGE4_JSON" \
  --route-csv "$ROUTE_CSV" \
  --device "$DEVICE" \
  --episodes "$EPISODES" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
