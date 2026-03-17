# DoorKey Extraction Report

## Current Conclusion

- `flat_dense` remains the strongest greedy DoorKey control.
- recovered `token_dense` (`ppo.ent_coef=0.001`) remains the canonical tokenized control.
- PPO-alone `SARE` still loses the fair greedy DoorKey comparison.
- The claim that is now alive is still teacher-guided, but it is stronger inside DoorKey than it was in the claim-hardening phase:
  - teacher-logit `KL` learner-state supervision remains the only bounded method that turns `SARE` into a strong greedy DoorKey policy
  - on the fresh matched-control DoorKey lane, `KL` learner-state `token_dense` improves to mean greedy success `0.5417`, but `KL` learner-state `SARE` still stays at `1.0000`
  - across the combined original+fresh six-seed DoorKey picture, `KL` learner-state `SARE` reaches mean greedy success `0.8568` versus `0.6042` for matched `KL` learner-state `token_dense`
  - no `KL` learner-state `SARE` seed remains at greedy success `0.0`
  - causal routing perturbations on recovered seeds `7` and `23` drop greedy success from `1.0` to `0.0`
- The strengthened claim is still bounded:
  - it is a teacher-guided extraction result, not a PPO-only routed win
  - it is still DoorKey-only
  - the bounded KeyCorridor transfer check remains flat

## Final Decision Path

Source artifacts:

- [lss_claim_consolidation_reproduction_note.md](outputs/reports/lss_claim_consolidation_reproduction_note.md)
- [lss_fresh_matched_control_report.md](outputs/reports/lss_fresh_matched_control_report.md)
- [lss_combined_doorkey_report.md](outputs/reports/lss_combined_doorkey_report.md)
- [lss_causal_route_dependence_report.md](outputs/reports/lss_causal_route_dependence_report.md)
- [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md)
- [lss_claim_consolidation_decision_memo.md](outputs/reports/lss_claim_consolidation_decision_memo.md)

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

## What Changed In The Claim-Consolidation Phase

The current phase answers the next fairness-and-causality follow-up:

- Does the reopened DoorKey teacher-guided `SARE` result still hold under fresh matched teacher-guided tokenized controls?
- Does the recovered policy actually depend on routing choices rather than only having non-collapsed routing statistics?

The answer is:

- yes on DoorKey, under the same teacher-guided `KL` learner-state method
- yes under causal route-dependence probes
- no on bounded KeyCorridor transfer

On the fresh matched-control DoorKey seeds `23`, `29`, and `31`:

| Seed | recovered `token_dense` | KL learner-state `token_dense` | baseline PPO `SARE` | KL learner-state `SARE` |
| --- | ---: | ---: | ---: | ---: |
| `23` | `0.0000` | `0.0000` | `0.0000` | `1.0000` |
| `29` | `0.0000` | `0.6250` | `0.0000` | `1.0000` |
| `31` | `1.0000` | `1.0000` | `0.0000` | `1.0000` |

Mean greedy success:

- recovered `token_dense`: `0.3333`
- `KL` learner-state `token_dense`: `0.5417`
- `KL` learner-state `SARE`: `1.0000`

Across the combined original+fresh six-seed DoorKey picture:

| Variant | Mean Greedy Success | Complete-Seed Failures |
| --- | ---: | ---: |
| recovered `token_dense` | `0.4505` | `3` |
| `KL` learner-state `token_dense` | `0.6042` | `2` |
| baseline PPO `SARE` | `0.0000` | `6` |
| `KL` learner-state `SARE` | `0.8568` | `0` |

So the right claim is now stronger than a method-only statement, but still bounded:

- teacher-guided KL learner-state supervision helps both structured students
- on both the original and fresh DoorKey matched lanes, `SARE` keeps the higher mean than the matched teacher-guided tokenized control
- across all six DoorKey seeds, `SARE` also eliminates complete-seed failure while matched `token_dense` does not
- under this teacher-guided setting, the evidence now supports a routed DoorKey edge

## Route Integrity

The newly recovered DoorKey seed `23` still looks routed, and the current phase adds direct causal evidence that the recovered policy depends on routing.

Source artifacts:

- [lss_new_case_route_integrity_report.md](outputs/reports/lss_new_case_route_integrity_report.md)
- [lss_causal_route_dependence_report.md](outputs/reports/lss_causal_route_dependence_report.md)

Key result:

- baseline PPO `SARE` on seed `23`: greedy success `0.0000`, route entropy `1.3857`, active compute `0.5000`
- KL learner-state `SARE` on seed `23`: greedy success `1.0000`, route entropy `1.3804`, active compute `0.5000`
- published improved seed `19` remains in the same routing regime: route entropy `1.3837`, active compute `0.5000`
- bounded causal probes on recovered seeds `7` and `23` are harsher than the route-integrity snapshot:
  - every single-expert ablation drops greedy success from `1.0000` to `0.0000`
  - fixed-router override drops greedy success from `1.0000` to `0.0000`
  - route randomization drops greedy success from `1.0000` to `0.0000`

So the strengthened DoorKey result is not just statistically non-collapsed routing. Under the bounded probe family used here, recovered performance is causally routing-dependent.

## Transfer Check

The exact same method does not transfer under the bounded KeyCorridor check.

Source artifact: [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md)

All three KeyCorridor seeds stayed flat:

- recovered `token_dense`: `0.0000`, `0.0000`, `0.0000`
- baseline PPO `SARE`: `0.0000`, `0.0000`, `0.0000`
- KL learner-state `SARE`: `0.0000`, `0.0000`, `0.0000`

## Recommendation

- Broaden within DoorKey only.
- The DoorKey routed claim is now stronger because it survives fresh matched teacher-guided tokenized controls and the recovered policy fails under routing perturbation.
- Keep the scope explicit:
  - teacher-guided extraction only
  - DoorKey only
  - external `64`-episode evaluation only
- Do not broaden this into a PPO-only or cross-task routed advantage claim while the bounded KeyCorridor transfer check remains flat.
