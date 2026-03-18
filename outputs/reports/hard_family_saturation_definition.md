# Saturation-Scale Hard-Family Definition

- current candidate: `post_unlock_weighted`
- git commit: `21a890285831feb89c6de03ea16619239ff732e3`
- git dirty: `True`
- definition ready: `True`

## Family Definition

- The hard family is defined by DoorKey blocks where the thaw-qualified routed candidate still trails matched KL learner-state `token_dense` under the external 64-episode path and the remaining weakness is still a late-phase cleanup problem rather than a broad task failure.
- qualification threshold for new blocks: matched token_dense mean at least `0.75` and candidate-minus-token_dense at most `-0.05`.
- selected development blocks: `[{'lane': 'post_pass_b', 'seeds': [73, 79, 83]}, {'lane': 'post_pass_c', 'seeds': [89, 97, 101]}, {'lane': 'post_pass_f', 'seeds': [137, 139, 149]}]`
- selected holdout blocks: `[{'lane': 'fresh_final', 'seeds': [47, 53, 59]}, {'lane': 'post_pass_e', 'seeds': [113, 127, 131]}]`

## New Block Pool Summary

| Block | Candidate SARE | token_dense | single_expert | recovered token_dense | baseline PPO SARE | Candidate-token | Complete-Seed Failures | Qualified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| post_pass_d | 0.2135 | 0.6667 | 0.0521 | 0.6667 | 0.0000 | -0.4531 | 2 | `no` |
| post_pass_e | 0.6667 | 1.0000 | 0.5417 | 1.0000 | 0.0000 | -0.3333 | 1 | `yes` |
| post_pass_f | 0.5000 | 1.0000 | 0.5781 | 0.6667 | 0.0000 | -0.5000 | 1 | `yes` |

## Selected Current-Candidate Snapshot

| Block | Seed | Variant | Greedy Success |
| --- | --- | --- | ---: |
| fresh_final | 47 | KL learner-state SARE | 0.4531 |
| fresh_final | 47 | KL learner-state single_expert | 0.4531 |
| fresh_final | 47 | KL learner-state token_dense | 1.0000 |
| fresh_final | 53 | KL learner-state SARE | 0.5156 |
| fresh_final | 53 | KL learner-state single_expert | 0.5156 |
| fresh_final | 53 | KL learner-state token_dense | 1.0000 |
| fresh_final | 59 | KL learner-state SARE | 0.4219 |
| fresh_final | 59 | KL learner-state single_expert | 0.4219 |
| fresh_final | 59 | KL learner-state token_dense | 1.0000 |
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
