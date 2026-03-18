#!/usr/bin/env bash
set -euo pipefail

STAGE2_JSON="${PSMN_STAGE2_JSON:-outputs/reports/long_campaign_stage2_screening.json}"
STAGE3_JSON="${PSMN_STAGE3_JSON:-outputs/reports/long_campaign_stage3_fairness.json}"
STAGE4_JSON="${PSMN_STAGE4_JSON:-outputs/reports/long_campaign_stage4_replication.json}"
STAGE5_JSON="${PSMN_STAGE5_JSON:-outputs/reports/long_campaign_stage5_route_validation.json}"
GATE_JSON="${PSMN_GATE_JSON:-outputs/reports/long_campaign_gate_report.json}"
MEMO_OUTPUT="${PSMN_MEMO_OUTPUT:-outputs/reports/long_campaign_decision_memo.md}"

source .venv/bin/activate
args=(--stage2-json "$STAGE2_JSON" --output "$MEMO_OUTPUT")
[[ -f "$STAGE3_JSON" ]] && args+=(--stage3-json "$STAGE3_JSON")
[[ -f "$STAGE4_JSON" ]] && args+=(--stage4-json "$STAGE4_JSON")
[[ -f "$STAGE5_JSON" ]] && args+=(--stage5-json "$STAGE5_JSON")
[[ -f "$GATE_JSON" ]] && args+=(--gate-json "$GATE_JSON")

python -m psmn_rl.analysis.lss_long_campaign decision-memo "${args[@]}"
