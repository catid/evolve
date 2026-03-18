#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_hard_family_program/campaign.yaml}"

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
print(reports["stage5_json"])
print(campaign["stage_roots"]["stage3_fairness"])
print(campaign["stage_roots"]["stage4_holdout"])
print(campaign["stage_roots"]["stage5_antiregression"])
print(reports["stage6_route_raw"])
print(reports["stage6_route_raw_csv"])
print(reports["stage6_report"])
print(reports["stage6_csv"])
print(reports["stage6_json"])
for candidate, meta in campaign["candidates"].items():
    print(f"candidate:{candidate}:{meta.get('reuse_root', '')}")
PY
)

STAGE3_JSON="${cfg[0]}"
STAGE4_JSON="${cfg[1]}"
STAGE5_JSON="${cfg[2]}"
STAGE3_ROOT="${cfg[3]}"
STAGE4_ROOT="${cfg[4]}"
STAGE5_ROOT="${cfg[5]}"
ROUTE_RAW="${cfg[6]}"
ROUTE_CSV="${cfg[7]}"
REPORT_OUTPUT="${cfg[8]}"
REPORT_CSV="${cfg[9]}"
REPORT_JSON="${cfg[10]}"
ROWS=("${cfg[@]:11}")

declare -A CANDIDATE_REUSE_ROOT=()
for row in "${ROWS[@]}"; do
  candidate="${row#candidate:}"
  candidate="${candidate%%:*}"
  reuse="${row#candidate:${candidate}:}"
  CANDIDATE_REUSE_ROOT["$candidate"]="$reuse"
done

readarray -t cases < <(
  source .venv/bin/activate
  python - "$STAGE3_JSON" "$STAGE4_JSON" "$STAGE5_JSON" <<'PY'
import json
import sys
from pathlib import Path

stage3 = json.loads(Path(sys.argv[1]).read_text())
stage4 = json.loads(Path(sys.argv[2]).read_text())
stage5 = json.loads(Path(sys.argv[3]).read_text())
best = stage3.get("best_candidate")
if not best or not stage5.get("stage5_pass"):
    raise SystemExit(0)
for name, payload in [("dev", stage3.get("selected_dev_case")), ("holdout", stage4.get("selected_holdout_case")), ("healthy", stage5.get("selected_healthy_case"))]:
    if payload:
        print("\t".join([name, str(payload["lane"]), str(payload["seed"]), best]))
PY
)

if [[ "${#cases[@]}" -gt 0 ]]; then
  declare -a route_args=()
  for row in "${cases[@]}"; do
    IFS=$'\t' read -r kind lane seed candidate <<<"$row"
    case "$kind" in
      dev)
        if [[ -n "${CANDIDATE_REUSE_ROOT[$candidate]}" ]]; then
          run_dir="${CANDIDATE_REUSE_ROOT[$candidate]}/${lane}/seed_${seed}/kl_lss_sare"
        else
          run_dir="${STAGE3_ROOT}/${candidate}/${lane}/seed_${seed}/kl_lss_sare"
        fi
        ;;
      holdout)
        run_dir="${STAGE4_ROOT}/${candidate}/${lane}/seed_${seed}/kl_lss_sare"
        ;;
      healthy)
        run_dir="${STAGE5_ROOT}/${candidate}/${lane}/seed_${seed}/kl_lss_sare"
        ;;
    esac
    route_args+=(--case "$lane" "$seed" "$run_dir")
  done
  python -m psmn_rl.analysis.lss_route_dependence \
    "${route_args[@]}" \
    --episodes "$EPISODES" \
    --device "$DEVICE" \
    --output "$ROUTE_RAW" \
    --csv "$ROUTE_CSV"
else
  mkdir -p "$(dirname "$ROUTE_RAW")" "$(dirname "$ROUTE_CSV")"
  printf '# Hard-Family Stage 6 Route Raw\n\n- skipped: `candidate did not survive anti-regression`\n' >"$ROUTE_RAW"
  : >"$ROUTE_CSV"
fi

python -m psmn_rl.analysis.lss_hard_family_program stage6-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage3-json "$STAGE3_JSON" \
  --stage4-json "$STAGE4_JSON" \
  --stage5-json "$STAGE5_JSON" \
  --route-csv "$ROUTE_CSV" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
