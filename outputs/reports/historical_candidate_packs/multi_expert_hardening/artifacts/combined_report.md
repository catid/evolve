# multi_expert_hardening Combined Adapter Report

- lane/seed coverage: `[('fresh', 23), ('fresh', 29), ('fresh', 31), ('fresh_extra', 37), ('fresh_extra', 41), ('fresh_extra', 43), ('fresh_final', 47), ('fresh_final', 53), ('fresh_final', 59), ('original', 7), ('original', 11), ('original', 19)]`

| Variant | Mean | Min | Max | Complete Seed Failures | Seed Count |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_sare` | `0.0000` | `0.0000` | `0.0000` | `12` | `12` |
| `kl_lss_sare` | `0.7122` | `0.0000` | `1.0000` | `1` | `12` |
| `kl_lss_single_expert` | `0.7604` | `0.0000` | `1.0000` | `1` | `9` |
| `kl_lss_token_dense` | `0.6354` | `0.0000` | `1.0000` | `4` | `12` |
| `recovered_token_dense` | `0.5586` | `0.0000` | `1.0000` | `5` | `12` |
