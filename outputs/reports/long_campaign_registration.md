# Long Campaign Registration

- frozen manifest: `configs/claims/doorkey_frozen_claim.yaml`
- git commit: `e9f3257e660833253ed59f4c59923f5184139347`
- git dirty: `True`

## Intervention Family Shortlist

- `cap_recent_4096`: a more moderate recent-step cap can suppress stale failed append-all accumulation without fully erasing useful late rounds
- `phase_balanced_recent_4096`: capping by phase with extra post-unlock quota can rebalance weak-block late-phase coverage without a new extraction family
- `post_unlock_weighted`: heavier KL on post-unlock states can target the shared late collapse on 53 and 59 while leaving earlier phases intact
- `phase_balanced_recent_4096_post_unlock_weighted`: combining moderate phase-balanced recency with post-unlock KL emphasis can address both route-fragile seed 47 and the shared post-unlock failures on 53 and 59

## Seed Blocks

- weak block: `fresh_final` seeds `[47, 53, 59]`
- original block: `[7, 11, 19]`
- fresh block: `[23, 29, 31]`
- fresh-extra block: `[37, 41, 43]`

## Stage Gates

- Stage 2: weak-block mean must beat `0.3125` and show a non-noisy per-seed improvement pattern.
- Stage 3: candidate `SARE` must at least match same-block `single_expert` and not improve controls more strongly than routed `SARE`.
- Stage 4: combined DoorKey `SARE` mean must stay at or above `0.7122` with no new complete-seed failures on previously healthy blocks.
- Stage 5: fixed-router override and worst expert ablation must remain materially harmful on one improved weak seed and one strong seed.
- Stage 6: candidate pack must clear the existing pack-based gate with no narrative override.
