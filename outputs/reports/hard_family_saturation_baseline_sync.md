# Saturation-Scale Hard-Family Baseline Sync

- frozen benchmark pack: `outputs/reports/frozen_benchmark_pack.json`
- current thaw-qualified candidate pack: `outputs/reports/post_pass_candidate_pack.json`
- git commit: `21a890285831feb89c6de03ea16619239ff732e3`
- git dirty: `True`

## Frozen Gate Thresholds

- retry-block KL learner-state `SARE` mean to beat: `0.3125`
- retry-block KL learner-state `single_expert` mean to match or beat: `0.4635`
- combined KL learner-state `SARE` mean to preserve: `0.7122`

## Current Candidate Snapshot

- retry-block KL learner-state `SARE` mean: `0.4635`
- retry-block KL learner-state `single_expert` mean: `0.4635`
- frozen-comparable combined KL learner-state `SARE` mean: `0.7500`
- dev-family KL learner-state `SARE` mean: `0.5625` vs matched token_dense `0.9635`
- holdout KL learner-state `SARE` mean: `0.5651` vs matched token_dense `1.0000`

## Current Candidate Per-Seed Snapshot

| Lane | Seed | Variant | Greedy Success |
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
| post_pass_c | 89 | KL learner-state SARE | 0.0000 |
| post_pass_c | 89 | KL learner-state single_expert | 0.0000 |
| post_pass_c | 89 | KL learner-state token_dense | 1.0000 |
| post_pass_c | 97 | KL learner-state SARE | 0.0000 |
| post_pass_c | 97 | KL learner-state single_expert | 0.0000 |
| post_pass_c | 97 | KL learner-state token_dense | 1.0000 |
| post_pass_c | 101 | KL learner-state SARE | 1.0000 |
| post_pass_c | 101 | KL learner-state single_expert | 0.6719 |
| post_pass_c | 101 | KL learner-state token_dense | 0.6719 |
| post_pass_e | 113 | KL learner-state SARE | 1.0000 |
| post_pass_e | 113 | KL learner-state single_expert | 0.6250 |
| post_pass_e | 113 | KL learner-state token_dense | 1.0000 |
| post_pass_e | 127 | KL learner-state SARE | 0.0000 |
| post_pass_e | 127 | KL learner-state single_expert | 1.0000 |
| post_pass_e | 127 | KL learner-state token_dense | 1.0000 |
| post_pass_e | 131 | KL learner-state SARE | 1.0000 |
| post_pass_e | 131 | KL learner-state single_expert | 0.0000 |
| post_pass_e | 131 | KL learner-state token_dense | 1.0000 |
| post_pass_f | 137 | KL learner-state SARE | 1.0000 |
| post_pass_f | 137 | KL learner-state single_expert | 0.5781 |
| post_pass_f | 137 | KL learner-state token_dense | 1.0000 |
| post_pass_f | 139 | KL learner-state SARE | 0.5000 |
| post_pass_f | 139 | KL learner-state single_expert | 0.5625 |
| post_pass_f | 139 | KL learner-state token_dense | 1.0000 |
| post_pass_f | 149 | KL learner-state SARE | 0.0000 |
| post_pass_f | 149 | KL learner-state single_expert | 0.5938 |
| post_pass_f | 149 | KL learner-state token_dense | 1.0000 |
