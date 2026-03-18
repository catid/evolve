#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_successor_mega_league/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
holdout = json.loads(Path(campaign["reports"]["stage3_json"]).read_text())
anti_regression = json.loads(Path(campaign["reports"]["stage4_json"]).read_text())
raw_dir = Path(campaign["reports"]["route_raw_dir"])
raw_dir.mkdir(parents=True, exist_ok=True)
print(campaign["reports"]["stage5_report"])
print(campaign["reports"]["stage5_csv"])
print(campaign["reports"]["stage5_json"])
print(raw_dir)
print(campaign["current_canonical_name"])
best = holdout.get("best_candidate") if anti_regression.get("challenger_pass") else None
print(best or "")
print(campaign["stage_roots"]["stage1_screening"])
print(campaign["stage_roots"]["stage3_holdout"])
print(campaign["stage_roots"]["stage4_antiregression"])
for case_name in ("dev", "holdout", "healthy"):
    case = campaign["route_cases"][case_name]
    round6_root = campaign["current_round6_sare_roots"][case["lane"]]
    print(f"case:{case_name}:{case['lane']}:{case['seed']}:{round6_root}")
PY
)

REPORT_OUTPUT="${cfg[0]}"
REPORT_CSV="${cfg[1]}"
REPORT_JSON="${cfg[2]}"
RAW_DIR="${cfg[3]}"
ROUND6="${cfg[4]}"
BEST_CANDIDATE="${cfg[5]}"
STAGE1_ROOT="${cfg[6]}"
STAGE4_HOLDOUT_ROOT="${cfg[7]}"
STAGE5_ANTI_ROOT="${cfg[8]}"
ROWS=("${cfg[@]:9}")

run_route_case() {
  local lane="$1"
  local seed="$2"
  local run_dir="$3"
  local md="$4"
  local csv="$5"
  python -m psmn_rl.analysis.lss_route_dependence \
    --case "$lane" "$seed" "$run_dir" \
    --episodes "$EPISODES" \
    --device "$DEVICE" \
    --output "$md" \
    --csv "$csv"
}

for row in "${ROWS[@]}"; do
  case_name="${row#case:}"
  case_name="${case_name%%:*}"
  rest="${row#case:${case_name}:}"
  lane="${rest%%:*}"
  rest="${rest#${lane}:}"
  seed="${rest%%:*}"
  round6_root="${rest#${seed}:}"
  round6_dir="${round6_root}/seed_${seed}/kl_lss_sare"
  case "$case_name" in
    dev)
      challenger_dir="${STAGE1_ROOT}/${BEST_CANDIDATE}/${lane}/seed_${seed}/kl_lss_sare"
      ;;
    holdout)
      challenger_dir="${STAGE4_HOLDOUT_ROOT}/${BEST_CANDIDATE}/${lane}/seed_${seed}/kl_lss_sare"
      ;;
    healthy)
      challenger_dir="${STAGE5_ANTI_ROOT}/${BEST_CANDIDATE}/${lane}/seed_${seed}/kl_lss_sare"
      ;;
  esac

  run_route_case "$lane" "$seed" "$round6_dir" \
    "${RAW_DIR}/${ROUND6}_${case_name}.md" \
    "${RAW_DIR}/${ROUND6}_${case_name}.csv"

  if [[ -n "$BEST_CANDIDATE" ]]; then
    run_route_case "$lane" "$seed" "$challenger_dir" \
      "${RAW_DIR}/${BEST_CANDIDATE}_${case_name}.md" \
      "${RAW_DIR}/${BEST_CANDIDATE}_${case_name}.csv"
  fi
done

python -m psmn_rl.analysis.lss_successor_migration stage5-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
