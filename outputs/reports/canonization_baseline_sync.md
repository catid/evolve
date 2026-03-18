# Canonization Baseline Sync

- frozen benchmark pack: `outputs/reports/frozen_benchmark_pack.json`
- current thaw-qualified candidate pack: `outputs/reports/post_pass_candidate_pack.json`
- git commit: `d278b1c2ff0f75af6cc5e459091af98e4e5b751f`
- git dirty: `True`

## Frozen Thresholds

- retry-block KL learner-state `SARE` mean to beat: `0.3125`
- retry-block KL learner-state `single_expert` mean to match or beat: `0.4635`
- combined KL learner-state `SARE` mean to preserve: `0.7122`

## Current Thaw-Qualified Candidate Snapshot

- retry-block KL learner-state `SARE` mean: `0.4635`
- retry-block KL learner-state `single_expert` mean: `0.4635`
- frozen-comparable combined KL learner-state `SARE` mean: `0.7500`
- frozen pack schema version: `1`

## Current Post-PASS Blocks

| Block | Seed | Variant | Greedy Success |
| --- | --- | --- | ---: |
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

## Interpretation

- The accepted starting point remains the thaw-qualified `post_unlock_weighted` candidate relative to the frozen benchmark pack.
- The structural canonization blocker is unchanged at the start of this campaign: `post_pass_b` still leaves matched `token_dense` above the candidate even though the candidate clears the frozen-pack gate.
