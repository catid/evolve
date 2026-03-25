#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_deadlock_escape_program/campaign.yaml}"

source .venv/bin/activate
if python - "$CAMPAIGN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
payload = json.loads(Path(campaign["reports"]["rescue_stage3_json"]).read_text(encoding="utf-8"))
raise SystemExit(0 if bool(payload.get("bounded_rescue_justified")) else 1)
PY
then
  PSMN_CAMPAIGN_CONFIG="$CAMPAIGN_CONFIG" bash ./scripts/run_lss_expansion_mega_program_stage2.sh
  python -m psmn_rl.analysis.lss_portfolio_campaign verification-report \
    --campaign-config "$CAMPAIGN_CONFIG" \
    --output "$(./.venv/bin/python - <<'PY' "$CAMPAIGN_CONFIG"
import sys
from pathlib import Path
from psmn_rl.analysis.campaign_config import load_campaign_config
campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["stage2_verification_report"])
PY
)" \
    --csv "$(./.venv/bin/python - <<'PY' "$CAMPAIGN_CONFIG"
import sys
from pathlib import Path
from psmn_rl.analysis.campaign_config import load_campaign_config
campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["stage2_verification_csv"])
PY
)" \
    --json "$(./.venv/bin/python - <<'PY' "$CAMPAIGN_CONFIG"
import sys
from pathlib import Path
from psmn_rl.analysis.campaign_config import load_campaign_config
campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["stage2_verification_json"])
PY
)"
  python -m psmn_rl.analysis.lss_portfolio_campaign fairness-report \
    --campaign-config "$CAMPAIGN_CONFIG" \
    --output "$(./.venv/bin/python - <<'PY' "$CAMPAIGN_CONFIG"
import sys
from pathlib import Path
from psmn_rl.analysis.campaign_config import load_campaign_config
campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["stage2_report"])
PY
)" \
    --csv "$(./.venv/bin/python - <<'PY' "$CAMPAIGN_CONFIG"
import sys
from pathlib import Path
from psmn_rl.analysis.campaign_config import load_campaign_config
campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["stage2_csv"])
PY
)" \
    --json "$(./.venv/bin/python - <<'PY' "$CAMPAIGN_CONFIG"
import sys
from pathlib import Path
from psmn_rl.analysis.campaign_config import load_campaign_config
campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["stage2_json"])
PY
)"
  PSMN_CAMPAIGN_CONFIG="$CAMPAIGN_CONFIG" bash ./scripts/run_lss_expansion_mega_program_stage4.sh
  PSMN_CAMPAIGN_CONFIG="$CAMPAIGN_CONFIG" bash ./scripts/run_lss_expansion_mega_program_stage5.sh
fi

readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path

from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["reports"]["practicalization_stage2_report"])
print(campaign["reports"]["practicalization_stage2_json"])
PY
)

REPORT_OUTPUT="${cfg[0]}"
REPORT_JSON="${cfg[1]}"

python -m psmn_rl.analysis.lss_deadlock_escape practicalization-verification-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --json "$REPORT_JSON"
