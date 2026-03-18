# Successor Stress Stage 1 Screening

- predecessor mean: `0.3281`
- current canonical `round6` mean: `0.8333`
- best challenger: `round7`
- selected prospective route case: `{'lane': 'prospective_j', 'seed': 307, 'baseline_candidate': 'round6'}`

| Candidate | Mean Prospective SARE | Complete-Seed Failures |
| --- | ---: | ---: |
| `round6` | `0.8333` | `1` |
| `round7` | `0.8333` | `1` |
| `post_unlock_weighted` | `0.3281` | `3` |

| Candidate | Block | Seed | Variant | Greedy Success |
| --- | --- | --- | --- | ---: |
| `baseline` | prospective_i | 281 | baseline PPO SARE | 0.0000 |
| `baseline` | prospective_i | 281 | recovered token_dense | 0.0000 |
| `baseline` | prospective_i | 283 | baseline PPO SARE | 0.0000 |
| `baseline` | prospective_i | 283 | recovered token_dense | 1.0000 |
| `baseline` | prospective_i | 293 | baseline PPO SARE | 0.0000 |
| `baseline` | prospective_i | 293 | recovered token_dense | 0.0000 |
| `baseline` | prospective_j | 307 | baseline PPO SARE | 0.0000 |
| `baseline` | prospective_j | 307 | recovered token_dense | 0.0000 |
| `baseline` | prospective_j | 311 | baseline PPO SARE | 0.0000 |
| `baseline` | prospective_j | 311 | recovered token_dense | 1.0000 |
| `baseline` | prospective_j | 313 | baseline PPO SARE | 0.0000 |
| `baseline` | prospective_j | 313 | recovered token_dense | 0.0000 |
| `post_unlock_weighted` | prospective_i | 281 | KL learner-state SARE | 0.0000 |
| `post_unlock_weighted` | prospective_i | 283 | KL learner-state SARE | 0.0000 |
| `post_unlock_weighted` | prospective_i | 293 | KL learner-state SARE | 0.0000 |
| `post_unlock_weighted` | prospective_j | 307 | KL learner-state SARE | 1.0000 |
| `post_unlock_weighted` | prospective_j | 311 | KL learner-state SARE | 0.5156 |
| `post_unlock_weighted` | prospective_j | 313 | KL learner-state SARE | 0.4531 |
| `round6` | prospective_i | 281 | KL learner-state SARE | 1.0000 |
| `round6` | prospective_i | 283 | KL learner-state SARE | 0.0000 |
| `round6` | prospective_i | 293 | KL learner-state SARE | 1.0000 |
| `round6` | prospective_j | 307 | KL learner-state SARE | 1.0000 |
| `round6` | prospective_j | 311 | KL learner-state SARE | 1.0000 |
| `round6` | prospective_j | 313 | KL learner-state SARE | 1.0000 |
| `round7` | prospective_i | 281 | KL learner-state SARE | 1.0000 |
| `round7` | prospective_i | 283 | KL learner-state SARE | 0.0000 |
| `round7` | prospective_i | 293 | KL learner-state SARE | 1.0000 |
| `round7` | prospective_j | 307 | KL learner-state SARE | 1.0000 |
| `round7` | prospective_j | 311 | KL learner-state SARE | 1.0000 |
| `round7` | prospective_j | 313 | KL learner-state SARE | 1.0000 |
