# Long Campaign Retry-Block Report

- candidate: `post_unlock_weighted`

| Seed | recovered token_dense | KL learner-state token_dense | KL learner-state single_expert | baseline PPO SARE | KL learner-state SARE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 47 | 1.0 | 1.0 | 0.453125 | 0.0 | 0.453125 |
| 53 | 1.0 | 1.0 | 0.515625 | 0.0 | 0.515625 |
| 59 | 1.0 | 1.0 | 0.421875 | 0.0 | 0.421875 |

## Summary

| Variant | Mean | Complete-Seed Failures |
| --- | ---: | ---: |
| recovered token_dense | `1.0000` | `0` |
| KL learner-state token_dense | `1.0000` | `0` |
| KL learner-state single_expert | `0.4635` | `0` |
| baseline PPO SARE | `0.0000` | `3` |
| KL learner-state SARE | `0.4635` | `0` |
