#!/usr/bin/env bash
set -euo pipefail

bash ./scripts/run_lss_hard_family_register.sh
bash ./scripts/run_lss_hard_family_stage2.sh
bash ./scripts/run_lss_hard_family_stage3.sh
bash ./scripts/run_lss_hard_family_stage4.sh
bash ./scripts/run_lss_hard_family_stage5.sh
bash ./scripts/run_lss_hard_family_stage6.sh
bash ./scripts/run_lss_hard_family_stage7.sh
bash ./scripts/run_lss_hard_family_stage8.sh
