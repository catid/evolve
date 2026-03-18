#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
STAGE3_JSON="${PSMN_STAGE3_JSON:-outputs/reports/long_campaign_stage3_fairness.json}"
STAGE4_JSON="${PSMN_STAGE4_JSON:-outputs/reports/long_campaign_stage4_replication.json}"
STAGE3_ROOT="${PSMN_STAGE3_ROOT:-outputs/experiments/lss_long_campaign/stage3}"
STAGE4_ROOT="${PSMN_STAGE4_ROOT:-outputs/experiments/lss_long_campaign/stage4}"
ROUTE_OUTPUT="${PSMN_ROUTE_OUTPUT:-outputs/reports/long_campaign_stage5_route_validation_raw.md}"
ROUTE_CSV="${PSMN_ROUTE_CSV:-outputs/reports/long_campaign_stage5_route_validation.csv}"
REPORT_OUTPUT="${PSMN_REPORT_OUTPUT:-outputs/reports/long_campaign_stage5_route_validation.md}"
REPORT_JSON="${PSMN_REPORT_JSON:-outputs/reports/long_campaign_stage5_route_validation.json}"

source .venv/bin/activate
readarray -t route_args < <(
  python - "$STAGE3_JSON" "$STAGE4_JSON" "$STAGE3_ROOT" "$STAGE4_ROOT" <<'PY'
import json
import sys
from pathlib import Path

stage3 = json.loads(Path(sys.argv[1]).read_text())
stage4 = json.loads(Path(sys.argv[2]).read_text())
candidate = stage3.get("best_candidate")
if not candidate or not stage4.get("stage4_pass"):
    raise SystemExit(0)
weak = stage4.get("selected_weak_case") or {"lane": "fresh_final", "seed": 47}
strong = stage4.get("selected_strong_case") or {"lane": "original", "seed": 7}
stage3_root = Path(sys.argv[3])
stage4_root = Path(sys.argv[4])
weak_run = stage3_root / candidate / weak["lane"] / f"seed_{weak['seed']}" / "kl_lss_sare"
strong_run = stage4_root / candidate / strong["lane"] / f"seed_{strong['seed']}" / "kl_lss_sare"
print(candidate)
print(weak["lane"])
print(str(weak["seed"]))
print(str(weak_run))
print(strong["lane"])
print(str(strong["seed"]))
print(str(strong_run))
PY
)

if [[ "${#route_args[@]}" -eq 0 ]]; then
  python - "$REPORT_OUTPUT" "$REPORT_JSON" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text("# Long Campaign Stage 5 Route Validation\n\n## Status\n\n- skipped: `no stage-4 survivor`\n", encoding="utf-8")
Path(sys.argv[2]).write_text(json.dumps({"stage": "stage5", "candidate": None, "summaries": [], "stage5_pass": False, "skipped": True}, indent=2, sort_keys=True), encoding="utf-8")
PY
  exit 0
fi

candidate="${route_args[0]}"
weak_lane="${route_args[1]}"
weak_seed="${route_args[2]}"
weak_run="${route_args[3]}"
strong_lane="${route_args[4]}"
strong_seed="${route_args[5]}"
strong_run="${route_args[6]}"

python -m psmn_rl.analysis.lss_route_dependence \
  --case "$weak_lane" "$weak_seed" "$weak_run" \
  --case "$strong_lane" "$strong_seed" "$strong_run" \
  --episodes 64 \
  --device "$DEVICE" \
  --output "$ROUTE_OUTPUT" \
  --csv "$ROUTE_CSV"

python -m psmn_rl.analysis.lss_long_campaign stage5-report \
  --candidate "$candidate" \
  --route-csv "$ROUTE_CSV" \
  --output "$REPORT_OUTPUT" \
  --json "$REPORT_JSON"
