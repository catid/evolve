#!/usr/bin/env bash
set -euo pipefail

bash ./scripts/run_lss_successor_challenge_register.sh
bash ./scripts/run_lss_successor_challenge_stage1.sh
bash ./scripts/run_lss_successor_challenge_stage2.sh
bash ./scripts/run_lss_successor_challenge_stage3.sh
