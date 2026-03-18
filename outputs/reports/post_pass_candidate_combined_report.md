# Post-PASS Candidate Frozen-Comparable Combined DoorKey Report

- candidate: `post_unlock_weighted`
- scope: `frozen-comparable combined lane/seed set only`
- expanded qualification report: `outputs/reports/post_pass_stage2_full_fairness.md`

| Lane | Seed | Variant | Greedy Success |
| --- | --- | --- | ---: |
| fresh | 23 | baseline PPO SARE | 0.0000 |
| fresh | 23 | KL learner-state SARE | 1.0000 |
| fresh | 23 | KL learner-state single_expert | 0.4062 |
| fresh | 23 | KL learner-state token_dense | 0.0000 |
| fresh | 23 | recovered token_dense | 0.0000 |
| fresh | 29 | baseline PPO SARE | 0.0000 |
| fresh | 29 | KL learner-state SARE | 1.0000 |
| fresh | 29 | KL learner-state single_expert | 1.0000 |
| fresh | 29 | KL learner-state token_dense | 0.6250 |
| fresh | 29 | recovered token_dense | 0.0000 |
| fresh | 31 | baseline PPO SARE | 0.0000 |
| fresh | 31 | KL learner-state SARE | 1.0000 |
| fresh | 31 | KL learner-state single_expert | 1.0000 |
| fresh | 31 | KL learner-state token_dense | 1.0000 |
| fresh | 31 | recovered token_dense | 1.0000 |
| fresh_extra | 37 | baseline PPO SARE | 0.0000 |
| fresh_extra | 37 | KL learner-state SARE | 1.0000 |
| fresh_extra | 37 | KL learner-state single_expert | 1.0000 |
| fresh_extra | 37 | KL learner-state token_dense | 1.0000 |
| fresh_extra | 37 | recovered token_dense | 1.0000 |
| fresh_extra | 41 | baseline PPO SARE | 0.0000 |
| fresh_extra | 41 | KL learner-state SARE | 1.0000 |
| fresh_extra | 41 | KL learner-state single_expert | 0.4375 |
| fresh_extra | 41 | KL learner-state token_dense | 0.0000 |
| fresh_extra | 41 | recovered token_dense | 0.0000 |
| fresh_extra | 43 | baseline PPO SARE | 0.0000 |
| fresh_extra | 43 | KL learner-state SARE | 0.4688 |
| fresh_extra | 43 | KL learner-state single_expert | 1.0000 |
| fresh_extra | 43 | KL learner-state token_dense | 0.0000 |
| fresh_extra | 43 | recovered token_dense | 0.0000 |
| fresh_final | 47 | baseline PPO SARE | 0.0000 |
| fresh_final | 47 | KL learner-state SARE | 0.4531 |
| fresh_final | 47 | KL learner-state single_expert | 0.4531 |
| fresh_final | 47 | KL learner-state token_dense | 1.0000 |
| fresh_final | 47 | recovered token_dense | 1.0000 |
| fresh_final | 53 | baseline PPO SARE | 0.0000 |
| fresh_final | 53 | KL learner-state SARE | 0.5156 |
| fresh_final | 53 | KL learner-state single_expert | 0.5156 |
| fresh_final | 53 | KL learner-state token_dense | 1.0000 |
| fresh_final | 53 | recovered token_dense | 1.0000 |
| fresh_final | 59 | baseline PPO SARE | 0.0000 |
| fresh_final | 59 | KL learner-state SARE | 0.4219 |
| fresh_final | 59 | KL learner-state single_expert | 0.4219 |
| fresh_final | 59 | KL learner-state token_dense | 1.0000 |
| fresh_final | 59 | recovered token_dense | 1.0000 |
| original | 7 | baseline PPO SARE | 0.0000 |
| original | 7 | KL learner-state SARE | 1.0000 |
| original | 7 | KL learner-state single_expert | 1.0000 |
| original | 7 | KL learner-state token_dense | 1.0000 |
| original | 7 | recovered token_dense | 0.7031 |
| original | 11 | baseline PPO SARE | 0.0000 |
| original | 11 | KL learner-state SARE | 0.5625 |
| original | 11 | KL learner-state single_expert | 1.0000 |
| original | 11 | KL learner-state token_dense | 0.0000 |
| original | 11 | recovered token_dense | 0.0000 |
| original | 19 | baseline PPO SARE | 0.0000 |
| original | 19 | KL learner-state SARE | 0.5781 |
| original | 19 | KL learner-state single_expert | 0.0000 |
| original | 19 | KL learner-state token_dense | 1.0000 |
| original | 19 | recovered token_dense | 1.0000 |

## Summary

| Variant | Mean | Complete-Seed Failures |
| --- | ---: | ---: |
| recovered token_dense | `0.5586` | `5` |
| KL learner-state token_dense | `0.6354` | `4` |
| KL learner-state single_expert | `0.6862` | `1` |
| baseline PPO SARE | `0.0000` | `12` |
| KL learner-state SARE | `0.7500` | `0` |
