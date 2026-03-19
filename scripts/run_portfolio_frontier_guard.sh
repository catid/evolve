#!/usr/bin/env bash
set -euo pipefail

./.venv/bin/python -m pytest -q \
  tests/test_portfolio_frontier_contract_snapshot.py \
  tests/test_portfolio_frontier_contract_loader.py \
  tests/test_portfolio_frontier_doctor.py \
  tests/test_portfolio_frontier_docs_audit.py \
  tests/test_portfolio_frontier_guard_workflow.py \
  tests/test_portfolio_frontier_guard_report.py

./.venv/bin/python -m psmn_rl.analysis.portfolio_frontier_doctor \
  --output outputs/reports/portfolio_frontier_doctor.md \
  --json outputs/reports/portfolio_frontier_doctor.json \
  --fail-on-drift

./.venv/bin/python -m psmn_rl.analysis.portfolio_frontier_docs_audit \
  --output outputs/reports/portfolio_frontier_docs_audit.md \
  --json outputs/reports/portfolio_frontier_docs_audit.json \
  --fail-on-drift

./.venv/bin/python -m psmn_rl.analysis.portfolio_frontier_guard_report \
  --output outputs/reports/portfolio_frontier_guard_report.md \
  --json outputs/reports/portfolio_frontier_guard_report.json \
  --fail-on-drift
