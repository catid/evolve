# Hard-Family Campaign Registration

- frozen benchmark pack: `outputs/reports/frozen_benchmark_pack.json`
- current thaw-qualified candidate pack: `outputs/reports/post_pass_candidate_pack.json`
- current candidate: `post_unlock_weighted`
- git commit: `adb5d321cb47166796fd17e87efee8d6cb027c64`
- git dirty: `True`

## Splits

- dev blocks: `[{'lane': 'post_pass_b', 'seeds': [73, 79, 83]}, {'lane': 'post_pass_c', 'seeds': [89, 97, 101]}]`
- holdout blocks: `[{'lane': 'fresh_final', 'seeds': [47, 53, 59]}]`
- anti-regression healthy blocks: `[{'lane': 'original', 'seeds': [7, 11, 19]}, {'lane': 'fresh', 'seeds': [23, 29, 31]}, {'lane': 'fresh_extra', 'seeds': [37, 41, 43]}, {'lane': 'post_pass_a', 'seeds': [61, 67, 71]}]`

## Stage Gates

- Stage 2: the candidate must materially improve over the thaw-qualified incumbent on the hard-family dev split, narrow the matched token_dense gap, and avoid new complete-seed failures.
- Stage 3: on the hard-family dev split, `SARE` must no longer trail matched `token_dense` and must remain at least competitive with matched `single_expert`.
- Stage 4: the best dev survivor must generalize that gain to the hard-family holdout without falling back to the original token-dense-led pattern.
- Stage 5: the holdout survivor must not materially worsen the healthier DoorKey blocks or lose thaw-qualified status.
- Stage 6: route probes on dev, holdout, and healthy cases must still show meaningful routing dependence.
- Stage 7: the improved candidate must avoid narrow checkpoint spikes on dev, holdout, and healthy cases.
- Stage 8: the successor candidate pack must validate, clear the pack-based gate relative to the frozen benchmark, and only then decide canonical vs thaw-qualified vs fallback.

## Reports

- family definition: `outputs/reports/hard_family_definition.md`
- baseline sync: `outputs/reports/hard_family_baseline_sync.md`
- shortlist: `outputs/reports/hard_family_stage1_shortlist.md`
- dev screening: `outputs/reports/hard_family_stage2_dev_screening.md`
- fairness: `outputs/reports/hard_family_stage3_fairness.md`
- holdout: `outputs/reports/hard_family_stage4_holdout.md`
- anti-regression: `outputs/reports/hard_family_stage5_antiregression.md`
- route validation: `outputs/reports/hard_family_stage6_route_validation.md`
- stability: `outputs/reports/hard_family_stage7_stability.md`
- successor pack: `outputs/reports/hard_family_successor_candidate_pack.json`
- decision memo: `outputs/reports/hard_family_canonization_decision_memo.md`
