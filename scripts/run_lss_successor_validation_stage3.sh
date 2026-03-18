#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
EPISODES="${PSMN_EVAL_EPISODES:-64}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_successor_validation/campaign.yaml}"

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
best_case = None
best_key = None
for row in stage1["rows"]:
    if row["candidate"] != selected or row["label"] != "kl_lss_sare":
        continue
    incumbent = next(
        (
            float(other["final_greedy_success"])
            for other in stage1["rows"]
            if other["candidate"] == campaign["incumbent_candidate_name"]
            and other["label"] == "kl_lss_sare"
            and other["lane"] == row["lane"]
            and int(other["seed"]) == int(row["seed"])
        ),
        0.0,
    )
    key = (
        float(row["final_greedy_success"]) - incumbent,
        float(row["final_greedy_success"]),
        str(row["lane"]),
        -int(row["seed"]),
    )
    if best_key is None or key > best_key:
        best_key = key
        best_case = {"lane": str(row["lane"]), "seed": int(row["seed"])}
if best_case is None:
    raise SystemExit("no route case available for selected candidate")
stage1_root = Path(campaign["stage_roots"]["stage1_screening"])
run_dir = stage1_root / selected / best_case["lane"] / f"seed_{best_case['seed']}" / "kl_lss_sare"
print(best_case["lane"])
print(best_case["seed"])
print(run_dir)
print(campaign["reports"]["route_raw_markdown"])
print(campaign["reports"]["route_raw_csv"])
print(campaign["reports"]["route_report"])
print(campaign["reports"]["route_csv"])
print(campaign["reports"]["route_json"])
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
ROUTE_REPORT="${cfg[5]}"
ROUTE_REPORT_CSV="${cfg[6]}"
ROUTE_REPORT_JSON="${cfg[7]}"
DECISION_MEMO="${cfg[8]}"
STAGE1_JSON="${cfg[9]}"
STAGE2_JSON="${cfg[10]}"

python -m psmn_rl.analysis.lss_route_dependence \
  --case "$LANE" "$SEED" "$RUN_DIR" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --output "$ROUTE_RAW_MD" \
  --csv "$ROUTE_RAW_CSV"

python -m psmn_rl.analysis.lss_successor_validation route-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --route-csv "$ROUTE_RAW_CSV" \
  --output "$ROUTE_REPORT" \
  --csv "$ROUTE_REPORT_CSV" \
  --json "$ROUTE_REPORT_JSON"

python -m psmn_rl.analysis.lss_successor_validation decision-memo \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage1-json "$STAGE1_JSON" \
  --stage2-json "$STAGE2_JSON" \
  --route-json "$ROUTE_REPORT_JSON" \
  --output "$DECISION_MEMO"
