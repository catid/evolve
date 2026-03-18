# Long Campaign Stage 4 Replication

- candidate: `post_unlock_weighted`

| Lane | Seed | Variant | Final Greedy | Best Round | Best Round Greedy |
| --- | --- | --- | ---: | ---: | ---: |
| fresh | 23 | KL learner-state SARE | 1.0000 | 4 | 1.0000 |
| fresh | 23 | KL learner-state single_expert | 0.4062 | 4 | 0.4062 |
| fresh | 23 | KL learner-state token_dense | 0.0000 | 1 | 0.0000 |
| fresh | 29 | KL learner-state SARE | 1.0000 | 4 | 1.0000 |
| fresh | 29 | KL learner-state single_expert | 1.0000 | 4 | 1.0000 |
| fresh | 29 | KL learner-state token_dense | 0.6250 | 4 | 0.6250 |
| fresh | 31 | KL learner-state SARE | 1.0000 | 4 | 1.0000 |
| fresh | 31 | KL learner-state single_expert | 1.0000 | 4 | 1.0000 |
| fresh | 31 | KL learner-state token_dense | 1.0000 | 1 | 1.0000 |
| fresh_extra | 37 | KL learner-state SARE | 1.0000 | 4 | 1.0000 |
| fresh_extra | 37 | KL learner-state single_expert | 1.0000 | 4 | 1.0000 |
| fresh_extra | 37 | KL learner-state token_dense | 1.0000 | 1 | 1.0000 |
| fresh_extra | 41 | KL learner-state SARE | 1.0000 | 4 | 1.0000 |
| fresh_extra | 41 | KL learner-state single_expert | 0.4375 | 4 | 0.4375 |
| fresh_extra | 41 | KL learner-state token_dense | 0.0000 | 1 | 0.0000 |
| fresh_extra | 43 | KL learner-state SARE | 0.4688 | 4 | 0.4688 |
| fresh_extra | 43 | KL learner-state single_expert | 1.0000 | 4 | 1.0000 |
| fresh_extra | 43 | KL learner-state token_dense | 0.0000 | 1 | 0.0000 |
| fresh_final | 47 | KL learner-state SARE | 0.453125 | 4 | 0.453125 |
| fresh_final | 47 | KL learner-state single_expert | 0.453125 | 4 | 0.453125 |
| fresh_final | 47 | KL learner-state token_dense | 1.0 | 1 | 1.0 |
| fresh_final | 53 | KL learner-state SARE | 0.515625 | 3 | 0.515625 |
| fresh_final | 53 | KL learner-state single_expert | 0.515625 | 3 | 0.515625 |
| fresh_final | 53 | KL learner-state token_dense | 1.0 | 1 | 1.0 |
| fresh_final | 59 | KL learner-state SARE | 0.421875 | 4 | 0.421875 |
| fresh_final | 59 | KL learner-state single_expert | 0.421875 | 4 | 0.421875 |
| fresh_final | 59 | KL learner-state token_dense | 1.0 | 1 | 1.0 |
| original | 7 | KL learner-state SARE | 1.0000 | 3 | 1.0000 |
| original | 7 | KL learner-state single_expert | 1.0000 | 4 | 1.0000 |
| original | 7 | KL learner-state token_dense | 1.0000 | 2 | 1.0000 |
| original | 11 | KL learner-state SARE | 0.5625 | 4 | 0.5625 |
| original | 11 | KL learner-state single_expert | 1.0000 | 4 | 1.0000 |
| original | 11 | KL learner-state token_dense | 0.0000 | 1 | 0.0000 |
| original | 19 | KL learner-state SARE | 0.5781 | 4 | 0.5781 |
| original | 19 | KL learner-state single_expert | 0.0000 | 1 | 0.0000 |
| original | 19 | KL learner-state token_dense | 1.0000 | 1 | 1.0000 |

## Candidate Combined Summary

| Variant | Mean Greedy Success | Complete-Seed Failures |
| --- | ---: | ---: |
| KL learner-state token_dense | `0.6354` | `4` |
| KL learner-state single_expert | `0.6862` | `1` |
| KL learner-state SARE | `0.7500` | `0` |

## Interpretation

- selected weak case for route validation: `(fresh_final, 53)`
- selected strong case for route validation: `(fresh, 23)`
- new complete-seed failures on previously healthy strong seeds: `[]`
- stage-4 status: `pass`
