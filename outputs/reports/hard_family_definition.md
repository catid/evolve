# Hard-Family Definition

- current candidate: `post_unlock_weighted`
- git commit: `adb5d321cb47166796fd17e87efee8d6cb027c64`
- git dirty: `True`

## Family Definition

- The hard family is defined by a shared late-phase DoorKey failure signature: the thaw-qualified routed candidate remains behind matched KL learner-state `token_dense` under the external 64-episode path, the gap is concentrated after unlock, and the block is hard without broadening beyond DoorKey.
- Development blocks stay on the explicit post-pass family introduced in the canonization lane: `post_pass_b` and `post_pass_c`.
- The holdout stays on an independent previously known hard block: `fresh_final`. It is withheld from intervention selection in this program and used only as the hard-family test split.

## Current Candidate Snapshot

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

## Family Membership Notes

- `post_pass_b`: retained as the existing canonization blocker because matched `token_dense` is still perfect while the current candidate stays below it.
- `post_pass_c`: kept in development because it was introduced as the same post-pass prime-seed family and shares the late cleanup imbalance targeted by the hard-block fixes.
- `fresh_final`: used as holdout because it is an independent historical hard DoorKey block where matched `token_dense` still dominates the current candidate and the late-phase routed failure story remains relevant.
