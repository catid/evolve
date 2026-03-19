#!/usr/bin/env bash
set -euo pipefail

bash ./scripts/run_lss_expansion_mega_program_register.sh
bash ./scripts/run_lss_expansion_mega_program_stage1.sh
bash ./scripts/run_lss_expansion_mega_program_stage2.sh
bash ./scripts/run_lss_expansion_mega_program_stage3.sh
bash ./scripts/run_lss_expansion_mega_program_stage4.sh
bash ./scripts/run_lss_expansion_mega_program_stage5.sh
bash ./scripts/run_lss_expansion_mega_program_stage6.sh
bash ./scripts/run_lss_expansion_mega_program_stage7.sh
bash ./scripts/run_lss_expansion_mega_program_stage8.sh
bash ./scripts/run_lss_expansion_mega_program_stage9.sh
