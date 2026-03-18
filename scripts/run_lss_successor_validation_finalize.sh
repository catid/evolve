#!/usr/bin/env bash
set -euo pipefail

bash ./scripts/run_lss_successor_validation_register.sh
bash ./scripts/run_lss_successor_validation_stage1.sh
bash ./scripts/run_lss_successor_validation_stage2.sh
bash ./scripts/run_lss_successor_validation_stage3.sh
