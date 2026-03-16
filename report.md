# DoorKey Extraction Report

## Current Conclusion

- `flat_dense` remains the strongest greedy DoorKey control.
- recovered `token_dense` (`ppo.ent_coef=0.001`) remains the canonical tokenized control.
- PPO-alone `SARE` still loses the fair greedy DoorKey comparison.
- The claim that has now changed is narrower and teacher-guided:
  - hard-label learner-state supervision was not robust enough
  - teacher-logit KL learner-state supervision is now the only bounded method that passes the repo’s external 3-seed gate for `SARE`
  - the reopened routed claim is a DoorKey extraction result, not a PPO result

## Final Decision Path

Source artifacts:

- [lss_robustness_reproduction_note.md](outputs/reports/lss_robustness_reproduction_note.md)
- [lss_seed_heterogeneity_report.md](outputs/reports/lss_seed_heterogeneity_report.md)
- [lss_robustness_sweep_report.md](outputs/reports/lss_robustness_sweep_report.md)
- [lss_robustness_multiseed_report.md](outputs/reports/lss_robustness_multiseed_report.md)
- [lss_route_integrity_report.md](outputs/reports/lss_route_integrity_report.md)
- [lss_robustness_decision_memo.md](outputs/reports/lss_robustness_decision_memo.md)

All final claims in this phase use the external `64`-episode `policy_diagnostics` path.

## What Still Stands Negative

Source artifacts:

- [checkpoint_dynamics_report.md](outputs/reports/checkpoint_dynamics_report.md)
- [entropy_schedule_report.md](outputs/reports/entropy_schedule_report.md)
- [self_imitation_report.md](outputs/reports/self_imitation_report.md)
- [policy_distillation_report.md](outputs/reports/policy_distillation_report.md)

These earlier no-go results still stand:

- checkpoint selection did not reveal a good greedy PPO `SARE` policy
- entropy schedules did not recover greedy PPO `SARE`
- self-imitation did not recover greedy PPO `SARE`
- offline teacher distillation did not recover greedy `SARE`

So the repo’s positive routed result does not come from PPO tuning or offline imitation.

## What Changed In The Robustness Phase

The fresh learner-state robustness phase answers a narrower question:

- Can routed `SARE` become a competent greedy DoorKey policy under a bounded teacher-guided extraction method?

The answer is now yes for one specific method family:

- `flat_dense` teacher
- learner-state supervision on student-visited states
- `kl` teacher-logit target
- `append_all` aggregation

The bounded robustness sweep shows why:

- switching from hard teacher actions to teacher-logit KL lifts seed `7` from greedy `0.5000` to `1.0000`
- it lifts previously dead seed `19` from greedy `0.0000` to `0.5781`
- capped replay variants (`cap_recent`, `cap_recent_balanced`) are both negative

## Multi-Seed Gate

Under the repo’s external 3-seed gate:

| Seed | recovered `token_dense` | baseline PPO `SARE` | KL learner-state `SARE` |
| --- | ---: | ---: | ---: |
| `7` | `0.7031` | `0.0000` | `1.0000` |
| `11` | `0.0000` | `0.0000` | `0.5625` |
| `19` | `1.0000` | `0.0000` | `0.5781` |

Mean greedy success:

- recovered `token_dense`: `0.5677`
- KL learner-state `SARE`: `0.7135`

So the improved routed method clears the repo bar:

- it beats recovered `token_dense` mean greedy success
- no seed remains at greedy `0.0`

## Route Integrity

The revived routed policy still looks routed on the critical seed `19`.

Source artifact: [lss_route_integrity_report.md](outputs/reports/lss_route_integrity_report.md)

Key result:

- baseline PPO `SARE`: greedy success `0.0000`, route entropy `1.3822`, active compute `0.5000`
- improved KL learner-state `SARE`: greedy success `0.5938`, route entropy `1.3837`, active compute `0.5000`

Expert loads remain balanced across all four experts, so the reopened claim is not an obvious route-collapse artifact.

## Recommendation

- Reopen routed greedy-performance work on DoorKey, but only under the narrower teacher-guided extraction claim backed by [lss_robustness_decision_memo.md](outputs/reports/lss_robustness_decision_memo.md).
- Do not reinterpret this as a PPO-only routed win.
- Keep the external multi-seed gate and route-integrity check as hard requirements before extending the claim to other tasks or architectures.
