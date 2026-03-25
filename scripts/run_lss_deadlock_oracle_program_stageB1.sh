#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_deadlock_oracle_program/campaign.yaml}"

ACTIVE_CAMPAIGN_CONFIG="$CAMPAIGN_CONFIG"
export CAMPAIGN_CONFIG

source .venv/bin/activate

if python - <<'PY'
from pathlib import Path
import json
import os

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(os.environ["CAMPAIGN_CONFIG"]))
payload = json.loads(Path(campaign["reports"]["oracle_synthesis_json"]).read_text(encoding="utf-8"))
raise SystemExit(0 if bool(payload.get("architecture_branch_justified")) else 1)
PY
then
  :
else
  ACTIVE_CAMPAIGN_CONFIG="outputs/experiments/lss_deadlock_oracle_program/runtime/no_archpilot_campaign.yaml"
  python - <<'PY'
from pathlib import Path
import json
import os

import yaml

from psmn_rl.analysis.campaign_config import load_campaign_config

source_path = Path(os.environ["CAMPAIGN_CONFIG"])
campaign = load_campaign_config(source_path)
filtered = {name: meta for name, meta in campaign["candidates"].items() if str(meta.get("track", "fruitful")) != "archpilot"}
campaign["candidates"] = filtered
campaign["candidate_subset"] = list(filtered)
target = Path("outputs/experiments/lss_deadlock_oracle_program/runtime/no_archpilot_campaign.yaml")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")

reason = (
    "Stage A6 synthesis did not justify the architecture-adjacent pilot, "
    "so the practical screen ran only the accepted-family fruitful and exploratory tracks."
)
report_path = Path(campaign["reports"]["stage1_archpilot_report"])
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(
    "\n".join(
        [
            "# Deadlock Oracle Stage B1 Architecture-Adjacent Screening",
            "",
            "- architecture branch justified: `False`",
            "- status: `skipped by oracle synthesis`",
            f"- reason: {reason}",
            "",
            "## Interpretation",
            "",
            "- The oracle program did not justify spending practical benchmark-lane budget on the quarantined architecture pilot in this run.",
        ]
    )
    + "\n",
    encoding="utf-8",
)
json_path = Path(campaign["reports"]["stage1_archpilot_json"])
json_path.write_text(
    json.dumps(
        {
            "architecture_branch_justified": False,
            "status": "skipped_by_oracle_synthesis",
            "advancing_candidates": [],
            "candidate_summaries": [],
            "rows": [],
            "reason": reason,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
fi

PSMN_CAMPAIGN_CONFIG="$ACTIVE_CAMPAIGN_CONFIG" bash ./scripts/run_lss_expansion_mega_program_stage1.sh

python -m psmn_rl.analysis.lss_portfolio_campaign stage1-screening \
  --campaign-config "$ACTIVE_CAMPAIGN_CONFIG"
