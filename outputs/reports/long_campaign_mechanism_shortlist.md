# Long Campaign Mechanism Shortlist

| Candidate | Intervention Family | Mechanism Hypothesis | Weak-Block Target | Expected Failure Mode |
| --- | --- | --- | --- | --- |
| `cap_recent_4096` | stale_append_all_cap | a more moderate recent-step cap can suppress stale failed append-all accumulation without fully erasing useful late rounds | 47, 53, 59 | underfits the healthy strong blocks by throwing away too much context |
| `phase_balanced_recent_4096` | phase_balanced_data_shaping | capping by phase with extra post-unlock quota can rebalance weak-block late-phase coverage without a new extraction family | 53, 59 | over-samples failing late states and keeps the same wrong action pattern |
| `post_unlock_weighted` | phase_aware_kl_weighting | heavier KL on post-unlock states can target the shared late collapse on 53 and 59 while leaving earlier phases intact | 53, 59 | helps all structured students equally and stays method-first rather than routed-first |
| `phase_balanced_recent_4096_post_unlock_weighted` | combined_phase_shaping | combining moderate phase-balanced recency with post-unlock KL emphasis can address both route-fragile seed 47 and the shared post-unlock failures on 53 and 59 | 47, 53, 59 | overfits the weak block and regresses the stronger historical seeds |
