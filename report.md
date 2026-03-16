# DoorKey Extraction Report

## Current Conclusion

- `flat_dense` remains the strongest greedy DoorKey control.
- recovered `token_dense` (`ppo.ent_coef=0.001`) remains the canonical tokenized control.
- PPO-alone `SARE` still loses the fair greedy DoorKey comparison.
- The claim that has now changed is narrower and teacher-guided:
  - hard-label learner-state supervision was not robust enough
  - teacher-logit KL learner-state supervision is now the only bounded method that passes the repo’s external 3-seed gate for `SARE`
  - the reopened routed claim is a DoorKey extraction result, not a PPO result
- The current claim-hardening phase strengthens that DoorKey result, but keeps it narrow:
  - KL learner-state `SARE` also wins on a fresh 3-seed DoorKey set
  - matched teacher-guided `token_dense` improves, but does not erase the DoorKey routed edge
  - the same method shows no bounded transfer to KeyCorridor

## Final Decision Path

Source artifacts:

- [lss_claim_hardening_reproduction_note.md](outputs/reports/lss_claim_hardening_reproduction_note.md)
- [lss_additional_seed_report.md](outputs/reports/lss_additional_seed_report.md)
- [lss_matched_control_report.md](outputs/reports/lss_matched_control_report.md)
- [lss_new_case_route_integrity_report.md](outputs/reports/lss_new_case_route_integrity_report.md)
- [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md)
- [lss_claim_hardening_decision_memo.md](outputs/reports/lss_claim_hardening_decision_memo.md)

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

## What Changed In The Claim-Hardening Phase

The current phase answers a narrower follow-up:

- Does the reopened DoorKey teacher-guided `SARE` result survive a fresh seed set and a matched teacher-guided tokenized-control comparison?

The answer is:

- yes on DoorKey, under the same teacher-guided KL learner-state method
- no on bounded KeyCorridor transfer

On the fresh DoorKey seeds `23`, `29`, and `31`:

| Seed | recovered `token_dense` | baseline PPO `SARE` | KL learner-state `SARE` |
| --- | ---: | ---: | ---: |
| `23` | `0.0000` | `0.0000` | `1.0000` |
| `29` | `0.0000` | `0.0000` | `1.0000` |
| `31` | `1.0000` | `0.0000` | `1.0000` |

Mean greedy success:

- recovered `token_dense`: `0.3333`
- KL learner-state `SARE`: `1.0000`

On the original seeds `7`, `11`, and `19`, matched teacher-guided extraction also helps `token_dense`:

| Seed | recovered `token_dense` | KL learner-state `token_dense` | KL learner-state `SARE` |
| --- | ---: | ---: | ---: |
| `7` | `0.7031` | `1.0000` | `1.0000` |
| `11` | `0.0000` | `0.0000` | `0.5625` |
| `19` | `1.0000` | `1.0000` | `0.5781` |

Mean greedy success:

- recovered `token_dense`: `0.5677`
- KL learner-state `token_dense`: `0.6667`
- KL learner-state `SARE`: `0.7135`

So the right claim is still method-first, but not routing-empty:

- teacher-guided KL learner-state supervision helps both structured students
- on the original DoorKey seeds, `SARE` still keeps the higher mean than the matched teacher-guided tokenized control
- on fresh DoorKey seeds, KL learner-state `SARE` avoids complete-seed failure again

## Route Integrity

The newly recovered DoorKey seed `23` still looks routed.

Source artifact: [lss_new_case_route_integrity_report.md](outputs/reports/lss_new_case_route_integrity_report.md)

Key result:

- baseline PPO `SARE` on seed `23`: greedy success `0.0000`, route entropy `1.3857`, active compute `0.5000`
- KL learner-state `SARE` on seed `23`: greedy success `1.0000`, route entropy `1.3804`, active compute `0.5000`
- published improved seed `19` remains in the same routing regime: route entropy `1.3837`, active compute `0.5000`

So the strengthened DoorKey result is still not an obvious route-collapse artifact.

## Transfer Check

The exact same method does not transfer under the bounded KeyCorridor check.

Source artifact: [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md)

All three KeyCorridor seeds stayed flat:

- recovered `token_dense`: `0.0000`, `0.0000`, `0.0000`
- baseline PPO `SARE`: `0.0000`, `0.0000`, `0.0000`
- KL learner-state `SARE`: `0.0000`, `0.0000`, `0.0000`

## Recommendation

- Keep routed work alive, but stay narrowly scoped.
- The hardened DoorKey claim now has stronger support, but it is still a teacher-guided extraction claim rather than a PPO-only routed win.
- Do not broaden this into a cross-task routed advantage claim while the bounded KeyCorridor transfer check remains flat.
