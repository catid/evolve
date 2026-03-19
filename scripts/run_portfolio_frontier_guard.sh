#!/usr/bin/env bash
set -euo pipefail

./.venv/bin/python -m pytest -q \
  tests/test_portfolio_frontier_contract_snapshot.py \
  tests/test_portfolio_frontier_contract_loader.py \
  tests/test_portfolio_frontier_doctor.py

./.venv/bin/python -m psmn_rl.analysis.portfolio_frontier_doctor \
  --output outputs/reports/portfolio_frontier_doctor.md \
  --json outputs/reports/portfolio_frontier_doctor.json \
  --fail-on-drift
