# Hard-Family Saturation Stage 4 Anti-Regression

- best candidate: `round6`
- current healthy-block KL learner-state `SARE` mean: `0.8841`
- candidate healthy-block KL learner-state `SARE` mean: `1.0000`

| Lane | Seed | Variant | Final Greedy | Best Round | Best-Round Greedy |
| --- | --- | --- | ---: | ---: | ---: |
| fresh | 23 | KL learner-state SARE | 1.0000 | 4 | 1.0000 |
| fresh | 23 | KL learner-state single_expert | 1.0000 | 5 | 1.0000 |
| fresh | 23 | KL learner-state token_dense | 0.5000 | 6 | 0.5000 |
| fresh | 29 | KL learner-state SARE | 1.0000 | 4 | 1.0000 |
| fresh | 29 | KL learner-state single_expert | 0.0000 | 4 | 1.0000 |
| fresh | 29 | KL learner-state token_dense | 1.0000 | 5 | 1.0000 |
| fresh | 31 | KL learner-state SARE | 1.0000 | 4 | 1.0000 |
| fresh | 31 | KL learner-state single_expert | 1.0000 | 4 | 1.0000 |
| fresh | 31 | KL learner-state token_dense | 1.0000 | 1 | 1.0000 |
| fresh_extra | 37 | KL learner-state SARE | 1.0000 | 4 | 1.0000 |
| fresh_extra | 37 | KL learner-state single_expert | 1.0000 | 4 | 1.0000 |
| fresh_extra | 37 | KL learner-state token_dense | 1.0000 | 1 | 1.0000 |
| fresh_extra | 41 | KL learner-state SARE | 1.0000 | 4 | 1.0000 |
| fresh_extra | 41 | KL learner-state single_expert | 1.0000 | 6 | 1.0000 |
| fresh_extra | 41 | KL learner-state token_dense | 1.0000 | 6 | 1.0000 |
| fresh_extra | 43 | KL learner-state SARE | 1.0000 | 5 | 1.0000 |
| fresh_extra | 43 | KL learner-state single_expert | 1.0000 | 4 | 1.0000 |
| fresh_extra | 43 | KL learner-state token_dense | 0.4688 | 6 | 0.4688 |
| original | 7 | KL learner-state SARE | 1.0000 | 3 | 1.0000 |
| original | 7 | KL learner-state single_expert | 1.0000 | 4 | 1.0000 |
| original | 7 | KL learner-state token_dense | 1.0000 | 2 | 1.0000 |
| original | 11 | KL learner-state SARE | 1.0000 | 5 | 1.0000 |
| original | 11 | KL learner-state single_expert | 1.0000 | 4 | 1.0000 |
| original | 11 | KL learner-state token_dense | 1.0000 | 5 | 1.0000 |
| original | 19 | KL learner-state SARE | 1.0000 | 5 | 1.0000 |
| original | 19 | KL learner-state single_expert | 1.0000 | 5 | 1.0000 |
| original | 19 | KL learner-state token_dense | 1.0000 | 1 | 1.0000 |
| post_pass_a | 61 | KL learner-state SARE | 1.0000 | 3 | 1.0000 |
| post_pass_a | 61 | KL learner-state single_expert | 1.0000 | 5 | 1.0000 |
| post_pass_a | 61 | KL learner-state token_dense | 1.0000 | 4 | 1.0000 |
| post_pass_a | 67 | KL learner-state SARE | 1.0000 | 4 | 1.0000 |
| post_pass_a | 67 | KL learner-state single_expert | 1.0000 | 5 | 1.0000 |
| post_pass_a | 67 | KL learner-state token_dense | 1.0000 | 1 | 1.0000 |
| post_pass_a | 71 | KL learner-state SARE | 1.0000 | 4 | 1.0000 |
| post_pass_a | 71 | KL learner-state single_expert | 1.0000 | 5 | 1.0000 |
| post_pass_a | 71 | KL learner-state token_dense | 1.0000 | 1 | 1.0000 |

## Per-Block Means

| Block | Candidate SARE | Candidate token_dense | Candidate single_expert | Candidate - token_dense | Candidate - single_expert |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| fresh | 1.0000 | 0.8333 | 0.6667 | 0.1667 | 0.3333 |
| fresh_extra | 1.0000 | 0.8229 | 1.0000 | 0.1771 | 0.0000 |
| post_pass_a | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## Interpretation

- new complete-seed failures on previously healthy blocks: `[]`
- selected healthy case for route validation: `{'lane': 'post_pass_a', 'seed': 67}`
- stage-4 status: `pass`
