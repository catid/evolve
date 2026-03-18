# Canonization Campaign Registration

- frozen benchmark pack: `outputs/reports/frozen_benchmark_pack.json`
- current thaw-qualified candidate pack: `outputs/reports/post_pass_candidate_pack.json`
- current candidate: `post_unlock_weighted`
- git commit: `d278b1c2ff0f75af6cc5e459091af98e4e5b751f`
- git dirty: `True`

## Hard-Block Family

- existing hard block: `post_pass_b` seeds `[73, 79, 83]`
- new hard block: `post_pass_c` seeds `[89, 97, 101]`
- `post_pass_c` uses the next unused three-prime fresh DoorKey block after `post_pass_b`, so the added block stays in the same seed-pattern class rather than broadening tasks.

## Stage Gates

- Stage 2: a new candidate must materially improve over the thaw-qualified incumbent on the hard-block family, narrow the token_dense gap, and add no new complete-seed failures.
- Stage 3: on the hard-block family, `SARE` must no longer trail matched `token_dense` and must remain at least competitive with matched `single_expert`.
- Stage 4: the best hard-block candidate must not materially regress the existing strong/recovered DoorKey blocks or introduce new complete-seed failures there.
- Stage 5: the selected hard-block win must remain routing-dependent and avoid narrow checkpoint-spike behavior.
- Stage 6: the successor pack must stay frozen-comparable, clear the existing pack-based gate, and be stronger than the current thaw-qualified pack rather than merely different.

## Reports

- registration: `outputs/reports/canonization_campaign_registration.md`
- baseline sync: `outputs/reports/canonization_baseline_sync.md`
- shortlist: `outputs/reports/canonization_stage1_mechanism_shortlist.md`
- screening: `outputs/reports/canonization_stage2_hard_block_screening.md`
- fairness: `outputs/reports/canonization_stage3_hard_block_fairness.md`
- anti-regression: `outputs/reports/canonization_stage4_antiregression.md`
- route/stability: `outputs/reports/canonization_stage5_stability_and_route.md`
- successor pack: `outputs/reports/canonization_successor_candidate_pack.md`
- canonization decision: `outputs/reports/canonization_decision_memo.md`
