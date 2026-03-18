#!/usr/bin/env bash
set -euo pipefail

./scripts/run_lss_post_pass_campaign_stage2.sh
./scripts/run_lss_post_pass_campaign_stage3.sh
./scripts/run_lss_post_pass_campaign_stage4.sh
./scripts/run_lss_post_pass_campaign_stage5.sh
./scripts/run_lss_post_pass_campaign_stage6.sh
