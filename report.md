# DoorKey Teacher-Extraction Report

## Current Conclusion

- `flat_dense` remains the strongest and most stable verified greedy DoorKey control.
- `token_dense` with `ppo.ent_coef=0.001` remains the canonical recovered tokenized DoorKey control.
- The bounded teacher-guided extraction phase is now complete:
  - offline policy distillation did not recover greedy `token_dense` or greedy `SARE`
  - learner-state supervision from a `flat_dense` teacher materially improved greedy `SARE`
  - the best recovered `SARE` seed retained non-collapsed routing
  - the recovery did not pass a 3-seed robustness check
- Current repo recommendation: pause routed greedy-policy claims on DoorKey. If work continues, it should be framed as an extraction-method project rather than as evidence that routed PPO is competitive by itself.

## Reproduced Baseline

Source artifacts:

- [teacher_extraction_reproduction_note.md](outputs/reports/teacher_extraction_reproduction_note.md)
- [outputs/reproductions/teacher_extraction_baseline/report.md](outputs/reproductions/teacher_extraction_baseline/report.md)

The accepted teacher-extraction starting point was reproduced on the current experiment lane with `seed=7`, `world_size=4`, and greedy plus sampled evaluation:

| Variant | Greedy Success | Best Sampled Success | Train Return |
| --- | ---: | ---: | ---: |
| `flat_dense` | `1.000` | `1.000` | `0.960` |
| recovered `token_dense` | `0.750` | `1.000` | `0.942` |
| `SARE` | `0.000` | `1.000` | `0.744` |

That reproduction was close enough to the published artifacts to justify forward work.

## Prior Negative Recovery Families Still Stand

Source artifacts:

- [checkpoint_dynamics_report.md](outputs/reports/checkpoint_dynamics_report.md)
- [entropy_schedule_report.md](outputs/reports/entropy_schedule_report.md)
- [self_imitation_report.md](outputs/reports/self_imitation_report.md)

The previous no-go result is still valid:

- checkpoint selection never revealed a good greedy `SARE` checkpoint
- bounded entropy schedules preserved sampled competence but did not recover greedy `SARE`
- self-imitation from successful sampled trajectories did not recover greedy `SARE`

That left teacher-guided extraction as the next clean capacity-vs-training-path discriminator.

## Offline Policy Distillation

Source artifact: [policy_distillation_report.md](outputs/reports/policy_distillation_report.md)

The minimal offline teacher-distillation path was negative for greedy recovery.

Key result:

- `flat_dense -> token_dense`: even full-model distillation raised best sampled success from `0.125` to `0.969`, but greedy success stayed `0.000`
- `flat_dense -> SARE`: head-only, last-shared, and full-model distillation all preserved strong sampled behavior, but greedy success stayed `0.000`
- `token_dense -> SARE`: sampled behavior stayed strong, but greedy success also stayed `0.000`

Interpretation:

- offline teacher data alone is not enough to recover greedy action ordering in this repo
- the failure is not unique to `SARE`; the same offline path also failed for the tokenized student sanity check

## Learner-State Supervision

Source artifacts:

- [learner_state_supervision_report.md](outputs/reports/learner_state_supervision_report.md)
- [teacher_extraction_sare_compare_64.md](outputs/reports/teacher_extraction_sare_compare_64.md)

The bounded learner-state supervision loop produced the first real positive routed signal:

- `flat_dense -> token_dense` remained at greedy success `0.000`
- `flat_dense -> SARE` moved from greedy success `0.000` to `0.500` on the original seed-7 lane
- the recovered seed-7 `SARE` became a sharp policy, not merely a softer sampled policy:
  - greedy max action probability rose from `0.4177` to `0.9920`
  - greedy top-1 vs top-2 margin rose from `0.4066` to `9.2367`

This is evidence that the routed student can represent a competent greedy DoorKey policy under stronger teacher supervision, at least on some seeds.

## Route Integrity

Source artifacts:

- [distilled_route_integrity_report.md](outputs/reports/distilled_route_integrity_report.md)
- [distilled_route_integrity_best_seed_report.md](outputs/reports/distilled_route_integrity_best_seed_report.md)

The positive learner-state `SARE` result did not look like obvious route collapse.

On the strongest recovered seed:

- baseline PPO `SARE`: greedy success `0.000`, route entropy `1.3832`, active compute `0.5000`
- learner-state supervised `SARE`: greedy success `1.000`, route entropy `1.3383`, active compute `0.5000`

Expert usage stayed distributed across all four experts, and active compute did not increase. So the best recovered policy still meaningfully used routing.

## Multi-Seed Validation

Source artifact: [teacher_extraction_multiseed_report.md](outputs/reports/teacher_extraction_multiseed_report.md)

The conditional positive signal did not survive cleanly as a robust method win.

Under a consistent 64-episode external evaluation:

| Seed | recovered `token_dense` | baseline `SARE` | learner-state `SARE` |
| --- | ---: | ---: | ---: |
| `7` | `0.703` | `0.000` | `0.500` |
| `11` | `0.000` | `0.000` | `1.000` |
| `19` | `1.000` | `0.000` | `0.000` |

Interpretation:

- learner-state supervision clearly beats baseline PPO `SARE`
- but the method is not reliable enough to reopen a routed greedy claim
- it helps strongly on two seeds and fails completely on one seed
- the recovered `token_dense` control is also seed-sensitive, but the routed result still does not meet a clean robustness bar

## Recommendation

- Keep `flat_dense` as the strongest verified greedy DoorKey control.
- Keep `token_dense` with `ppo.ent_coef=0.001` as the canonical recovered tokenized control.
- Treat learner-state supervision as evidence that `SARE` capacity is not the whole problem: the routed student can represent a strong greedy DoorKey policy under some teacher-guided conditions.
- Do not reopen routed greedy-performance claims from this repo on the basis of the current teacher-guided result, because the multi-seed check is not robust enough.
- If work continues, treat it as a bounded extraction-method project with multi-seed validation as a hard gate.
