#!/usr/bin/env bash
set -euo pipefail

bash ./scripts/run_lss_successor_stress_register.sh
bash ./scripts/run_lss_successor_stress_stage1.sh
bash ./scripts/run_lss_successor_stress_stage2.sh
bash ./scripts/run_lss_successor_stress_stage3.sh
