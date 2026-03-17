#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
python -m psmn_rl.analysis.lss_multi_expert_hardening decision-memo \
  --single-expert-csv outputs/reports/lss_fresh_single_expert_matched_control_report.csv \
  --forensics-csv outputs/reports/lss_seed29_route_randomization_forensics.csv \
  --broader-route-csv outputs/reports/lss_broader_route_dependence_report.csv \
  --final-block-csv outputs/reports/lss_final_fresh_seed_block_report.csv \
  --combined-csv outputs/reports/lss_final_combined_doorkey_report.csv \
  --output outputs/reports/lss_multi_expert_hardening_decision_memo.md
