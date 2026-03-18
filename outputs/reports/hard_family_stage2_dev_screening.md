# Hard-Family Stage 2 Dev Screening

- current dev-family KL learner-state `SARE` mean: `0.5938`
- current dev-family matched token_dense mean: `0.9453`
- current dev-family matched single_expert mean: `0.2057`

| Candidate | Block | Seed | Final Greedy | Δ vs Current SARE | Δ vs Current token_dense | Δ vs Current single_expert | Best Round | Best-Round Disagreement |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `post_unlock_weighted_disagreement075` | post_pass_b | 73 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 3 | 0.7274 |
| `post_unlock_weighted_disagreement075` | post_pass_b | 79 | 0.5625 | -0.4375 | -0.4375 | 0.5625 | 4 | 0.6294 |
| `post_unlock_weighted_disagreement075` | post_pass_b | 83 | 0.5625 | 0.0000 | -0.4375 | 0.0000 | 4 | 0.9796 |
| `post_unlock_weighted_disagreement075` | post_pass_c | 89 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 4 | 0.9237 |
| `post_unlock_weighted_disagreement075` | post_pass_c | 97 | 0.0000 | 0.0000 | -1.0000 | 0.0000 | 1 | 0.9954 |
| `post_unlock_weighted_disagreement075` | post_pass_c | 101 | 1.0000 | 0.0000 | 0.3281 | 0.3281 | 4 | 0.9668 |
| `post_unlock_weighted_round5` | post_pass_b | 73 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 3 | 0.7274 |
| `post_unlock_weighted_round5` | post_pass_b | 79 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 3 | 0.7038 |
| `post_unlock_weighted_round5` | post_pass_b | 83 | 0.5625 | 0.0000 | -0.4375 | 0.0000 | 4 | 0.9796 |
| `post_unlock_weighted_round5` | post_pass_c | 89 | 0.5156 | 0.5156 | -0.4844 | 0.5156 | 5 | 0.5413 |
| `post_unlock_weighted_round5` | post_pass_c | 97 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 5 | 0.9645 |
| `post_unlock_weighted_round5` | post_pass_c | 101 | 1.0000 | 0.0000 | 0.3281 | 0.3281 | 4 | 0.9668 |
| `post_unlock_weighted_round5_phase_balanced` | post_pass_b | 73 | 0.5312 | -0.4688 | -0.4688 | 0.5312 | 5 | 0.7077 |
| `post_unlock_weighted_round5_phase_balanced` | post_pass_b | 79 | 0.0000 | -1.0000 | -1.0000 | 0.0000 | 1 | 0.9961 |
| `post_unlock_weighted_round5_phase_balanced` | post_pass_b | 83 | 0.0000 | -0.5625 | -1.0000 | -0.5625 | 1 | 1.0000 |
| `post_unlock_weighted_round5_phase_balanced` | post_pass_c | 89 | 0.0000 | 0.0000 | -1.0000 | 0.0000 | 1 | 0.8113 |
| `post_unlock_weighted_round5_phase_balanced` | post_pass_c | 97 | 0.0000 | 0.0000 | -1.0000 | 0.0000 | 1 | 0.9954 |
| `post_unlock_weighted_round5_phase_balanced` | post_pass_c | 101 | 0.0000 | -1.0000 | -0.6719 | -0.6719 | 1 | 0.9980 |
| `post_unlock_weighted_round5_phase_balanced_disagreement050` | post_pass_b | 73 | 0.5469 | -0.4531 | -0.4531 | 0.5469 | 5 | 0.6431 |
| `post_unlock_weighted_round5_phase_balanced_disagreement050` | post_pass_b | 79 | 0.0000 | -1.0000 | -1.0000 | 0.0000 | 1 | 0.9961 |
| `post_unlock_weighted_round5_phase_balanced_disagreement050` | post_pass_b | 83 | 0.0000 | -0.5625 | -1.0000 | -0.5625 | 1 | 1.0000 |
| `post_unlock_weighted_round5_phase_balanced_disagreement050` | post_pass_c | 89 | 0.0000 | 0.0000 | -1.0000 | 0.0000 | 1 | 0.8113 |
| `post_unlock_weighted_round5_phase_balanced_disagreement050` | post_pass_c | 97 | 0.0000 | 0.0000 | -1.0000 | 0.0000 | 4 | 0.7459 |
| `post_unlock_weighted_round5_phase_balanced_disagreement050` | post_pass_c | 101 | 0.0000 | -1.0000 | -0.6719 | -0.6719 | 1 | 0.9980 |

## Candidate Summary

| Candidate | Dev Mean | Δ vs Current SARE | Gap Narrowing vs token_dense | Candidate - token_dense | Candidate - single_expert | Complete-Seed Failures | Max Block Δ vs Current | Stage 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `post_unlock_weighted_round5` | 0.8464 | 0.2526 | 0.2526 | -0.0990 | 0.6406 | 0 | 0.5052 | `pass` |
| `post_unlock_weighted_disagreement075` | 0.6875 | 0.0938 | 0.0938 | -0.2578 | 0.4818 | 1 | 0.3333 | `pass` |
| `post_unlock_weighted_round5_phase_balanced_disagreement050` | 0.0911 | -0.5026 | -0.5026 | -0.8542 | -0.1146 | 5 | -0.3333 | `stop` |
| `post_unlock_weighted_round5_phase_balanced` | 0.0885 | -0.5052 | -0.5052 | -0.8568 | -0.1172 | 5 | -0.3333 | `stop` |

## Interpretation

- advancing candidates: `['post_unlock_weighted_round5', 'post_unlock_weighted_disagreement075']`
- Stage 2 keeps only candidates that beat the thaw-qualified incumbent on the dev split, narrow the matched token_dense gap, and avoid new complete-seed failures.
