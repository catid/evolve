# Hard-Family Stage 1 Shortlist

| Candidate | Intervention Family | Mechanism Hypothesis | Hard-Family Target | Broader DoorKey Risk |
| --- | --- | --- | --- | --- |
| `post_unlock_weighted_round5` | cleanup_round_extension | one extra learner-state clean-up round may let the current post-unlock weighting finish the hard-family late cleanup cases without changing the routed architecture or extraction family | fixes the stalled late cleanup on post_pass_b seed 83 and the partial post_pass_c conversion on seed 89 | a fifth round could overfit late cleanup and create a checkpoint spike that does not transfer to holdout or healthy blocks |
| `post_unlock_weighted_disagreement075` | late_disagreement_bonus | a disagreement bonus on top of post-unlock weighting may target the specific hard-family states where routed SARE still disagrees with the teacher after door unlock | amplifies hard-family late disagreement pockets without changing the learner-state extraction family | disagreement-heavy weighting may destabilize the healthier blocks and produce new complete-seed failures |
| `post_unlock_weighted_round5_phase_balanced` | cleanup_round_plus_phase_balanced_replay | combining one extra clean-up round with phase-balanced recent replay can keep late post-unlock coverage high while suppressing stale append-all domination on the hard-family dev split | the post_pass_b/post_pass_c family keeps accumulating long failed late episodes, so a recent phase-balanced buffer may protect the good late states that the round-5 variant is trying to convert | phase-balanced replay may help the hard-family dev blocks while erasing useful early-phase context on healthy blocks |
| `post_unlock_weighted_round5_phase_balanced_disagreement050` | cleanup_round_phase_balanced_disagreement | a modest disagreement bonus on top of round-5 phase-balanced replay may emphasize the exact hard-family late states that still remain mismatched after phase balancing | combines stale-data control with late-phase teacher-student mismatch targeting on the same dev family | may overfocus on disagreement-heavy late states and hurt the broader DoorKey picture |

## Interpretation

- The shortlist stays inside the existing teacher-guided KL learner-state family.
- The new candidates only add phase-balanced recent replay and a modest disagreement bonus on top of the already strongest round-5 late-phase intervention; this is a bounded hard-family program rather than a broad search.
