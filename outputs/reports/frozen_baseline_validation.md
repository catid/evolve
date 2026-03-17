# Frozen Baseline Validation

- manifest: `configs/claims/doorkey_frozen_claim.yaml`
- git commit: `acf13060031180632c5db77f2025ef4dda5ceb04`
- git dirty: `True`

## Frozen Summary

| Variant | Combined Mean | Retry-Block Mean |
| --- | ---: | ---: |
| recovered token_dense | 0.5586 | 1.0000 |
| KL learner-state token_dense | 0.6354 | 1.0000 |
| KL learner-state single_expert | 0.6862 | 0.4635 |
| baseline PPO SARE | 0.0000 | 0.0000 |
| KL learner-state SARE | 0.7122 | 0.3125 |

## Validation Checks

| Section | Metric | Variant | Expected | Actual | Status |
| --- | --- | --- | --- | --- | --- |
| combined | mean | recovered_token_dense | `0.5586` | `0.5586` | `PASS` |
| combined | mean | kl_lss_token_dense | `0.6354` | `0.6354` | `PASS` |
| combined | mean | kl_lss_single_expert | `0.6862` | `0.6862` | `PASS` |
| combined | mean | baseline_sare | `0.0000` | `0.0000` | `PASS` |
| combined | mean | kl_lss_sare | `0.7122` | `0.7122` | `PASS` |
| combined | complete_seed_failures | recovered_token_dense | `5` | `5` | `PASS` |
| combined | complete_seed_failures | kl_lss_token_dense | `4` | `4` | `PASS` |
| combined | complete_seed_failures | kl_lss_single_expert | `1` | `1` | `PASS` |
| combined | complete_seed_failures | baseline_sare | `12` | `12` | `PASS` |
| combined | complete_seed_failures | kl_lss_sare | `1` | `1` | `PASS` |
| retry_block | mean | recovered_token_dense | `1.0000` | `1.0000` | `PASS` |
| retry_block | mean | kl_lss_token_dense | `1.0000` | `1.0000` | `PASS` |
| retry_block | mean | kl_lss_single_expert | `0.4635` | `0.4635` | `PASS` |
| retry_block | mean | baseline_sare | `0.0000` | `0.0000` | `PASS` |
| retry_block | mean | kl_lss_sare | `0.3125` | `0.3125` | `PASS` |
| retry_block | complete_seed_failures | recovered_token_dense | `0` | `0` | `PASS` |
| retry_block | complete_seed_failures | kl_lss_token_dense | `0` | `0` | `PASS` |
| retry_block | complete_seed_failures | kl_lss_single_expert | `0` | `0` | `PASS` |
| retry_block | complete_seed_failures | baseline_sare | `3` | `3` | `PASS` |
| retry_block | complete_seed_failures | kl_lss_sare | `1` | `1` | `PASS` |
| keycorridor_transfer | mean | recovered_token_dense | `0.0000` | `0.0000` | `PASS` |
| keycorridor_transfer | mean | baseline_sare | `0.0000` | `0.0000` | `PASS` |
| keycorridor_transfer | mean | kl_lss_sare | `0.0000` | `0.0000` | `PASS` |
| coverage | combined_lane_seed_set | - | `[('fresh', 23), ('fresh', 29), ('fresh', 31), ('fresh_extra', 37), ('fresh_extra', 41), ('fresh_extra', 43), ('fresh_final', 47), ('fresh_final', 53), ('fresh_final', 59), ('original', 7), ('original', 11), ('original', 19)]` | `[('fresh', 23), ('fresh', 29), ('fresh', 31), ('fresh_extra', 37), ('fresh_extra', 41), ('fresh_extra', 43), ('fresh_final', 47), ('fresh_final', 53), ('fresh_final', 59), ('original', 7), ('original', 11), ('original', 19)]` | `PASS` |
| coverage | retry_block_lane_seed_set | - | `[('fresh_final', 47), ('fresh_final', 53), ('fresh_final', 59)]` | `[('fresh_final', 47), ('fresh_final', 53), ('fresh_final', 59)]` | `PASS` |

## Verdict

PASS: frozen baseline validated
