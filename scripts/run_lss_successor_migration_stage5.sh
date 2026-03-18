#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_successor_migration/campaign.yaml}"

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
stage3 = json.loads(Path(campaign["reports"]["stage3_json"]).read_text())
stage4 = json.loads(Path(campaign["reports"]["stage4_json"]).read_text())
raw_dir = Path(campaign["reports"]["route_raw_dir"])
raw_dir.mkdir(parents=True, exist_ok=True)
print(campaign["reports"]["stage5_report"])
print(campaign["reports"]["stage5_csv"])
print(campaign["reports"]["stage5_json"])
print(raw_dir)
print(campaign["current_canonical_name"])
best = stage3.get("best_candidate") if stage4.get("challenger_pass") else None
print(best or "")
for case_name in ("dev", "holdout", "healthy"):
    case = campaign["route_cases"][case_name]
    print(f"case:{case_name}:{case['lane']}:{case['seed']}")
PY
)

REPORT_OUTPUT="${cfg[0]}"
REPORT_CSV="${cfg[1]}"
REPORT_JSON="${cfg[2]}"
RAW_DIR="${cfg[3]}"
ROUND6="${cfg[4]}"
BEST_CANDIDATE="${cfg[5]}"
ROWS=("${cfg[@]:6}")

run_route_case() {
  local line_name="$1"
  local lane="$2"
  local seed="$3"
  local run_dir="$4"
  local md="$5"
  local csv="$6"
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
  seed="${rest##*:}"

  case "$case_name" in
    dev)
      round6_dir="outputs/experiments/lss_successor_stress_extended/stage1_screening/round6/${lane}/seed_${seed}/kl_lss_sare"
      challenger_dir="outputs/experiments/lss_successor_migration/stage1_screening/${BEST_CANDIDATE}/${lane}/seed_${seed}/kl_lss_sare"
      ;;
    holdout)
      round6_dir="outputs/experiments/lss_successor_tiebreak/stage1_screening/round6/${lane}/seed_${seed}/kl_lss_sare"
      challenger_dir="outputs/experiments/lss_successor_migration/stage3_holdout/${BEST_CANDIDATE}/${lane}/seed_${seed}/kl_lss_sare"
      ;;
    healthy)
      round6_dir="outputs/experiments/lss_hard_family_saturation/stage4_antiregression/round6/${lane}/seed_${seed}/kl_lss_sare"
      challenger_dir="outputs/experiments/lss_successor_migration/stage4_antiregression/${BEST_CANDIDATE}/${lane}/seed_${seed}/kl_lss_sare"
      ;;
  esac

  run_route_case "$ROUND6" "$lane" "$seed" "$round6_dir" \
    "${RAW_DIR}/${ROUND6}_${case_name}.md" \
    "${RAW_DIR}/${ROUND6}_${case_name}.csv"

  if [[ -n "$BEST_CANDIDATE" ]]; then
    run_route_case "$BEST_CANDIDATE" "$lane" "$seed" "$challenger_dir" \
      "${RAW_DIR}/${BEST_CANDIDATE}_${case_name}.md" \
      "${RAW_DIR}/${BEST_CANDIDATE}_${case_name}.csv"
  fi
done

python -m psmn_rl.analysis.lss_successor_migration stage5-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
