# final_frozen_state Retry-Block Adapter Report

- lane/seed coverage: `[('fresh_final', 47), ('fresh_final', 53), ('fresh_final', 59)]`

| Variant | Mean | Min | Max | Complete Seed Failures | Seed Count |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_sare` | `0.0000` | `0.0000` | `0.0000` | `3` | `3` |
| `kl_lss_sare` | `0.3125` | `0.0000` | `0.5156` | `1` | `3` |
| `kl_lss_single_expert` | `0.4635` | `0.4219` | `0.5156` | `0` | `3` |
| `kl_lss_token_dense` | `1.0000` | `1.0000` | `1.0000` | `0` | `3` |
| `recovered_token_dense` | `1.0000` | `1.0000` | `1.0000` | `0` | `3` |
