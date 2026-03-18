#!/usr/bin/env bash
set -euo pipefail

bash ./scripts/run_lss_canonization_campaign_register.sh
bash ./scripts/run_lss_canonization_campaign_stage2.sh
bash ./scripts/run_lss_canonization_campaign_stage3.sh
bash ./scripts/run_lss_canonization_campaign_stage4.sh
bash ./scripts/run_lss_canonization_campaign_stage5.sh
bash ./scripts/run_lss_canonization_campaign_stage6.sh
