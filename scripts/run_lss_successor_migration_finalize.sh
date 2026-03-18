#!/usr/bin/env bash
set -euo pipefail

bash ./scripts/run_lss_successor_migration_register.sh
bash ./scripts/run_lss_successor_migration_stage1.sh
bash ./scripts/run_lss_successor_migration_stage2.sh
bash ./scripts/run_lss_successor_migration_stage3.sh
bash ./scripts/run_lss_successor_migration_stage4.sh
bash ./scripts/run_lss_successor_migration_stage5.sh
bash ./scripts/run_lss_successor_migration_stage6.sh
bash ./scripts/run_lss_successor_migration_stage7.sh
