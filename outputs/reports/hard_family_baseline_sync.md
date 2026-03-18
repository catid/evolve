# Hard-Family Baseline Sync

- frozen benchmark pack: `outputs/reports/frozen_benchmark_pack.json`
- current thaw-qualified candidate pack: `outputs/reports/post_pass_candidate_pack.json`
- git commit: `adb5d321cb47166796fd17e87efee8d6cb027c64`
- git dirty: `True`

## Frozen Gate Thresholds

- retry-block KL learner-state `SARE` mean to beat: `0.3125`
- retry-block KL learner-state `single_expert` mean to match or beat: `0.4635`
- combined KL learner-state `SARE` mean to preserve: `0.7122`

## Current Candidate Snapshot

- retry-block KL learner-state `SARE` mean: `0.4635`
- retry-block KL learner-state `single_expert` mean: `0.4635`
- frozen-comparable combined KL learner-state `SARE` mean: `0.7500`
- dev-family KL learner-state `SARE` mean: `0.5938` vs matched token_dense `0.9453`
- holdout KL learner-state `SARE` mean: `0.4635` vs matched token_dense `1.0000`
