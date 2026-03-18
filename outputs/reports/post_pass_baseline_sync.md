# Post-PASS Baseline Sync

- frozen benchmark pack: `outputs/reports/frozen_benchmark_pack.json`
- current thaw-qualified candidate pack: `outputs/reports/long_campaign_candidate_pack.json`
- git commit: `90f4ca1e3b9a572156e49d4af86d273a748cea43`
- git dirty: `True`

## Frozen Thresholds

- retry-block KL learner-state `SARE` mean to beat: `0.3125`
- retry-block KL learner-state `single_expert` mean to match or beat: `0.4635`
- combined KL learner-state `SARE` mean to preserve: `0.7122`

## Current Candidate Snapshot

- current retry-block KL learner-state `SARE` mean: `0.4635`
- current retry-block KL learner-state `single_expert` mean: `0.4635`
- current combined KL learner-state `SARE` mean: `0.7500`

## Interpretation

- The frozen benchmark pack remains the comparison unit for this phase.
- The `post_unlock_weighted` candidate is thaw-qualified relative to the frozen pack, but not yet canonicalized as a successor benchmark.
- frozen pack schema version: `1`
