# claim_hardening Combined Adapter Report

- lane/seed coverage: `[('fresh', 23), ('fresh', 29), ('fresh', 31), ('original', 7), ('original', 11), ('original', 19)]`

| Variant | Mean | Min | Max | Complete Seed Failures | Seed Count |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_sare` | `0.0000` | `0.0000` | `0.0000` | `6` | `6` |
| `flat_dense` | `1.0000` | `1.0000` | `1.0000` | `0` | `6` |
| `kl_lss_sare` | `0.8568` | `0.5625` | `1.0000` | `0` | `6` |
| `kl_lss_token_dense` | `0.6667` | `0.0000` | `1.0000` | `1` | `3` |
| `recovered_token_dense` | `0.4505` | `0.0000` | `1.0000` | `3` | `6` |
