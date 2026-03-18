# Post-PASS Stage 2 Full Fairness

- candidate: `post_unlock_weighted`

| Block | Seed | Variant | Greedy Success |
| --- | --- | --- | ---: |
| fresh | 23 | KL learner-state SARE | 1.0000 |
| fresh | 23 | KL learner-state single_expert | 0.4062 |
| fresh | 23 | KL learner-state token_dense | 0.0000 |
| fresh | 29 | KL learner-state SARE | 1.0000 |
| fresh | 29 | KL learner-state single_expert | 1.0000 |
| fresh | 29 | KL learner-state token_dense | 0.6250 |
| fresh | 31 | KL learner-state SARE | 1.0000 |
| fresh | 31 | KL learner-state single_expert | 1.0000 |
| fresh | 31 | KL learner-state token_dense | 1.0000 |
| fresh_extra | 37 | KL learner-state SARE | 1.0000 |
| fresh_extra | 37 | KL learner-state single_expert | 1.0000 |
| fresh_extra | 37 | KL learner-state token_dense | 1.0000 |
| fresh_extra | 41 | KL learner-state SARE | 1.0000 |
| fresh_extra | 41 | KL learner-state single_expert | 0.4375 |
| fresh_extra | 41 | KL learner-state token_dense | 0.0000 |
| fresh_extra | 43 | KL learner-state SARE | 0.4688 |
| fresh_extra | 43 | KL learner-state single_expert | 1.0000 |
| fresh_extra | 43 | KL learner-state token_dense | 0.0000 |
| fresh_final | 47 | KL learner-state SARE | 0.4531 |
| fresh_final | 47 | KL learner-state single_expert | 0.4531 |
| fresh_final | 47 | KL learner-state token_dense | 1.0000 |
| fresh_final | 53 | KL learner-state SARE | 0.5156 |
| fresh_final | 53 | KL learner-state single_expert | 0.5156 |
| fresh_final | 53 | KL learner-state token_dense | 1.0000 |
| fresh_final | 59 | KL learner-state SARE | 0.4219 |
| fresh_final | 59 | KL learner-state single_expert | 0.4219 |
| fresh_final | 59 | KL learner-state token_dense | 1.0000 |
| original | 7 | KL learner-state SARE | 1.0000 |
| original | 7 | KL learner-state single_expert | 1.0000 |
| original | 7 | KL learner-state token_dense | 1.0000 |
| original | 11 | KL learner-state SARE | 0.5625 |
| original | 11 | KL learner-state single_expert | 1.0000 |
| original | 11 | KL learner-state token_dense | 0.0000 |
| original | 19 | KL learner-state SARE | 0.5781 |
| original | 19 | KL learner-state single_expert | 0.0000 |
| original | 19 | KL learner-state token_dense | 1.0000 |
| post_pass_a | 61 | KL learner-state SARE | 1.0000 |
| post_pass_a | 61 | KL learner-state single_expert | 0.0000 |
| post_pass_a | 61 | KL learner-state token_dense | 1.0000 |
| post_pass_a | 67 | KL learner-state SARE | 1.0000 |
| post_pass_a | 67 | KL learner-state single_expert | 0.5781 |
| post_pass_a | 67 | KL learner-state token_dense | 1.0000 |
| post_pass_a | 71 | KL learner-state SARE | 1.0000 |
| post_pass_a | 71 | KL learner-state single_expert | 0.6250 |
| post_pass_a | 71 | KL learner-state token_dense | 0.9531 |
| post_pass_b | 73 | KL learner-state SARE | 1.0000 |
| post_pass_b | 73 | KL learner-state single_expert | 0.0000 |
| post_pass_b | 73 | KL learner-state token_dense | 1.0000 |
| post_pass_b | 79 | KL learner-state SARE | 1.0000 |
| post_pass_b | 79 | KL learner-state single_expert | 0.0000 |
| post_pass_b | 79 | KL learner-state token_dense | 1.0000 |
| post_pass_b | 83 | KL learner-state SARE | 0.5625 |
| post_pass_b | 83 | KL learner-state single_expert | 0.5625 |
| post_pass_b | 83 | KL learner-state token_dense | 1.0000 |

## Per-Block Means

| Block | candidate SARE | matched token_dense | matched single_expert | candidate - token_dense | candidate - single_expert | candidate complete-seed failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fresh | 1.0000 | 0.5417 | 0.8021 | 0.4583 | 0.1979 | 0 |
| fresh_extra | 0.8229 | 0.3333 | 0.8125 | 0.4896 | 0.0104 | 0 |
| fresh_final | 0.4635 | 1.0000 | 0.4635 | -0.5365 | 0.0000 | 0 |
| original | 0.7135 | 0.6667 | 0.6667 | 0.0469 | 0.0469 | 0 |
| post_pass_a | 1.0000 | 0.9844 | 0.4010 | 0.0156 | 0.5990 | 0 |
| post_pass_b | 0.8542 | 1.0000 | 0.1875 | -0.1458 | 0.6667 | 0 |

## Combined Means

- candidate KL learner-state `SARE`: `0.8090`
- matched KL learner-state `token_dense`: `0.7543`
- matched KL learner-state `single_expert`: `0.5556`

## Interpretation

- stage-2 competitiveness status: `pass`
- The expanded fairness view keeps the per-block structure visible so the qualification decision does not collapse into a single summary mean.
