#!/usr/bin/env bash
set -euo pipefail

bash ./scripts/run_lss_hard_family_saturation_register.sh
bash ./scripts/run_lss_hard_family_saturation_stage1.sh
bash ./scripts/run_lss_hard_family_saturation_stage2.sh
bash ./scripts/run_lss_hard_family_saturation_stage3.sh
bash ./scripts/run_lss_hard_family_saturation_stage4.sh
bash ./scripts/run_lss_hard_family_saturation_stage5.sh
bash ./scripts/run_lss_hard_family_saturation_stage6.sh
bash ./scripts/run_lss_hard_family_saturation_stage7.sh
