# Saturation-Scale Hard-Family Registration

- frozen benchmark pack: `outputs/reports/frozen_benchmark_pack.json`
- current thaw-qualified candidate pack: `outputs/reports/post_pass_candidate_pack.json`
- current candidate: `post_unlock_weighted`
- git commit: `21a890285831feb89c6de03ea16619239ff732e3`
- git dirty: `True`
- definition ready: `True`

## Splits

- hard-family development blocks: `[{'lane': 'post_pass_b', 'seeds': [73, 79, 83]}, {'lane': 'post_pass_c', 'seeds': [89, 97, 101]}, {'lane': 'post_pass_f', 'seeds': [137, 139, 149]}]`
- hard-family holdout blocks: `[{'lane': 'fresh_final', 'seeds': [47, 53, 59]}, {'lane': 'post_pass_e', 'seeds': [113, 127, 131]}]`
- anti-regression healthy blocks: `[{'lane': 'original', 'seeds': [7, 11, 19]}, {'lane': 'fresh', 'seeds': [23, 29, 31]}, {'lane': 'fresh_extra', 'seeds': [37, 41, 43]}, {'lane': 'post_pass_a', 'seeds': [61, 67, 71]}]`

## Mechanism Directions

- total directions: `10`
- total screened variants: `20`

## Stage Gates

- Stage 1: coarse screening advances only the top three to four candidates that materially improve over the incumbent on the hard-family dev split, avoid obvious new failures, and plausibly narrow the token_dense gap.
- Stage 2: fairness advances only the top one to two candidates that no longer clearly trail matched token_dense on the dev split and stay competitive with matched single_expert.
- Stage 3: holdout requires the hard-family gain to generalize to holdout without reverting to the original token-dense-led pattern.
- Stage 4: anti-regression requires the best holdout survivor to preserve the broader DoorKey picture on healthy blocks and stay thaw-qualified.
- Stage 5: route validation requires the surviving gain to remain meaningfully routed on dev, holdout, and healthy seeds.
- Stage 6: stability requires the gain to avoid narrow checkpoint spikes on hard-family and healthy seeds.
- Stage 7: the successor pack must validate, clear the frozen benchmark gate, and only then decide canonization vs thaw-qualified vs fallback.

## Reports

- family definition: `outputs/reports/hard_family_saturation_definition.md`
- baseline sync: `outputs/reports/hard_family_saturation_baseline_sync.md`
- shortlist: `outputs/reports/hard_family_saturation_stage1_shortlist.md`
- Stage 1 screening: `outputs/reports/hard_family_saturation_stage1_screening.md`
- Stage 2 fairness: `outputs/reports/hard_family_saturation_stage2_fairness.md`
- Stage 3 holdout: `outputs/reports/hard_family_saturation_stage3_holdout.md`
- Stage 4 anti-regression: `outputs/reports/hard_family_saturation_stage4_antiregression.md`
- Stage 5 route validation: `outputs/reports/hard_family_saturation_stage5_route_validation.md`
- Stage 6 stability: `outputs/reports/hard_family_saturation_stage6_stability.md`
- successor pack: `outputs/reports/hard_family_saturation_successor_pack.json`
- decision memo: `outputs/reports/hard_family_saturation_decision_memo.md`
