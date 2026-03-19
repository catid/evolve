#!/usr/bin/env bash
set -euo pipefail

./.venv/bin/python -m pytest -q \
  tests/test_portfolio_frontier_contract_snapshot.py \
  tests/test_portfolio_frontier_contract_loader.py \
  tests/test_portfolio_frontier_doctor.py \
  tests/test_portfolio_frontier_docs_audit.py \
  tests/test_portfolio_frontier_guard_workflow.py \
  tests/test_portfolio_frontier_guard_workflow_contract.py \
  tests/test_portfolio_frontier_guard_workflow_contract_loader.py \
  tests/test_portfolio_frontier_guard_report.py \
  tests/test_portfolio_frontier_guard_report_snapshot.py \
  tests/test_portfolio_frontier_guard_report_loader.py \
  tests/test_portfolio_frontier_doctor_loader.py \
  tests/test_portfolio_frontier_docs_audit_loader.py \
  tests/test_portfolio_frontier_consistency_loader.py \
  tests/test_portfolio_frontier_manifest_loader.py \
  tests/test_portfolio_frontier_schedule_loader.py \
  tests/test_portfolio_frontier_kit_loader.py \
  tests/test_portfolio_frontier_replay_loader.py \
  tests/test_portfolio_candidate_pack_loader.py \
  tests/test_portfolio_gate_report_loader.py \
  tests/test_portfolio_active_state_doctor.py \
  tests/test_portfolio_active_state_doctor_loader.py \
  tests/test_portfolio_operational_state.py \
  tests/test_portfolio_operational_state_loader.py \
  tests/test_portfolio_seed_pack_loader.py \
  tests/test_portfolio_seed_pack_validation_loader.py \
  tests/test_portfolio_seed_pack_scorer_loader.py \
  tests/test_portfolio_seed_pack_doctor.py \
  tests/test_portfolio_seed_pack_doctor_loader.py

./.venv/bin/python -m psmn_rl.analysis.portfolio_frontier_doctor \
  --output outputs/reports/portfolio_frontier_doctor.md \
  --json outputs/reports/portfolio_frontier_doctor.json \
  --fail-on-drift

./.venv/bin/python -m psmn_rl.analysis.portfolio_frontier_docs_audit \
  --output outputs/reports/portfolio_frontier_docs_audit.md \
  --json outputs/reports/portfolio_frontier_docs_audit.json \
  --fail-on-drift

./.venv/bin/python -m psmn_rl.analysis.portfolio_active_state_doctor \
  --output outputs/reports/portfolio_active_state_doctor.md \
  --json outputs/reports/portfolio_active_state_doctor.json \
  --fail-on-drift

./.venv/bin/python -m psmn_rl.analysis.portfolio_operational_state \
  --output outputs/reports/portfolio_operational_state.md \
  --json outputs/reports/portfolio_operational_state.json

./.venv/bin/python -m psmn_rl.analysis.portfolio_frontier_guard_workflow_contract \
  --output outputs/reports/portfolio_frontier_guard_workflow_contract.md \
  --json outputs/reports/portfolio_frontier_guard_workflow_contract.json \
  --fail-on-drift

./.venv/bin/python -m psmn_rl.analysis.portfolio_frontier_guard_report \
  --output outputs/reports/portfolio_frontier_guard_report.md \
  --json outputs/reports/portfolio_frontier_guard_report.json \
  --fail-on-drift

./.venv/bin/python -m psmn_rl.analysis.portfolio_seed_pack_doctor \
  --output outputs/reports/portfolio_seed_pack_doctor.md \
  --json outputs/reports/portfolio_seed_pack_doctor.json \
  --fail-on-drift
