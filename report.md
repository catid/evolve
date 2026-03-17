# DoorKey Extraction Report

## Current Conclusion

- `flat_dense` remains the strongest greedy DoorKey control.
- recovered `token_dense` (`ppo.ent_coef=0.001`) remains the canonical tokenized control.
- PPO-alone `SARE` still loses the fair greedy DoorKey comparison.
- The live positive claim is still teacher-guided, but it is now broader within DoorKey than it was in the claim-consolidation phase:
  - teacher-logit `KL` learner-state supervision remains the only bounded method that turns `SARE` into a strong greedy DoorKey policy
  - the missing matched `single_expert` control does not erase that routed DoorKey edge on the original `7/11/19` lane:
    - `KL` learner-state `single_expert` mean greedy success `0.6667`
    - `KL` learner-state `SARE` mean greedy success `0.7135`
  - one more fresh matched DoorKey block also stays positive for routed `SARE`:
    - fresh-extra seeds `37/41/43`: `KL` learner-state `token_dense` mean greedy success `0.3333`
    - fresh-extra seeds `37/41/43`: `KL` learner-state `SARE` mean greedy success `0.8229`
  - across the expanded nine-seed DoorKey picture, `KL` learner-state `SARE` reaches mean greedy success `0.8455` versus `0.5139` for matched `KL` learner-state `token_dense`
  - no `KL` learner-state `SARE` DoorKey seed remains at greedy success `0.0`
  - causal routing perturbations now extend beyond the original `7/23` demonstration:
    - expert ablation and fixed-router override still collapse or severely damage success on seeds `7`, `19`, `23`, and `29`
    - route randomization is still catastrophic on `7`, `19`, and `23`, but only weakly harmful on fresh seed `29`
- The strengthened claim is still bounded:
  - it is a teacher-guided extraction result, not a PPO-only routed win
  - it is still DoorKey-only
  - the bounded KeyCorridor transfer check remains flat

## Final Decision Path

Source artifacts:

- [lss_claim_broadening_reproduction_note.md](outputs/reports/lss_claim_broadening_reproduction_note.md)
- [lss_single_expert_matched_control_report.md](outputs/reports/lss_single_expert_matched_control_report.md)
- [lss_extended_route_dependence_report.md](outputs/reports/lss_extended_route_dependence_report.md)
- [lss_additional_fresh_seed_block_report.md](outputs/reports/lss_additional_fresh_seed_block_report.md)
- [lss_expanded_combined_doorkey_report.md](outputs/reports/lss_expanded_combined_doorkey_report.md)
- [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md)
- [lss_claim_broadening_decision_memo.md](outputs/reports/lss_claim_broadening_decision_memo.md)

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

## What Changed In The Claim-Broadening Phase

The current phase answers the next bounded DoorKey-only follow-up:

- Does the strengthened DoorKey teacher-guided `SARE` result survive the missing matched `single_expert` control?
- Does the causal route-dependence story extend beyond the original `7/23` demonstration?
- Does one more fresh matched DoorKey seed block keep the edge alive?

The answer is:

- yes on the original missing-control fairness lane
- mostly yes on broader causal routing dependence, with one weaker random-routing case on seed `29`
- yes on the additional fresh matched DoorKey seed block
- still no on bounded KeyCorridor transfer

On the original matched missing-control lane:

| Seed | recovered `token_dense` | KL learner-state `token_dense` | KL learner-state `single_expert` | baseline PPO `SARE` | KL learner-state `SARE` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `7` | `0.7031` | `1.0000` | `1.0000` | `0.0000` | `1.0000` |
| `11` | `0.0000` | `0.0000` | `1.0000` | `0.0000` | `0.5625` |
| `19` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `0.5781` |

Mean greedy success:

- `KL` learner-state `token_dense`: `0.6667`
- `KL` learner-state `single_expert`: `0.6667`
- `KL` learner-state `SARE`: `0.7135`

On the additional fresh matched DoorKey seeds `37`, `41`, and `43`:

| Seed | recovered `token_dense` | KL learner-state `token_dense` | baseline PPO `SARE` | KL learner-state `SARE` |
| --- | ---: | ---: | ---: | ---: |
| `37` | `1.0000` | `1.0000` | `0.0000` | `1.0000` |
| `41` | `0.0000` | `0.0000` | `0.0000` | `1.0000` |
| `43` | `0.0000` | `0.0000` | `0.0000` | `0.4688` |

Mean greedy success:

- recovered `token_dense`: `0.3333`
- `KL` learner-state `token_dense`: `0.3333`
- `KL` learner-state `SARE`: `0.8229`

Across the expanded nine-seed DoorKey picture:

| Variant | Mean Greedy Success | Complete-Seed Failures |
| --- | ---: | ---: |
| recovered `token_dense` | `0.4115` | `5` |
| `KL` learner-state `token_dense` | `0.5139` | `4` |
| `KL` learner-state `single_expert` | `0.6667` on the original 3-seed slice | `1` |
| baseline PPO `SARE` | `0.0000` | `9` |
| `KL` learner-state `SARE` | `0.8455` | `0` |

So the right claim is now broader within DoorKey, but still bounded:

- teacher-guided KL learner-state supervision helps structured students generally
- `SARE` still stays ahead of the matched teacher-guided tokenized control on the original lane, the first fresh lane, and the extra fresh lane
- `SARE` also stays ahead of the matched teacher-guided `single_expert` control on the original fairness slice
- causal routing dependence now extends beyond the original 2-seed demonstration, though the route-randomization probe is not equally strong on every recovered seed
- under this teacher-guided setting, the evidence now supports a broader within-DoorKey routed edge

## Route Integrity

The newly recovered DoorKey seed `23` still looks routed, and the current phase adds direct causal evidence that the recovered policy depends on routing.

Source artifacts:

- [lss_new_case_route_integrity_report.md](outputs/reports/lss_new_case_route_integrity_report.md)
- [lss_causal_route_dependence_report.md](outputs/reports/lss_causal_route_dependence_report.md)
- [lss_extended_route_dependence_report.md](outputs/reports/lss_extended_route_dependence_report.md)

Key result:

- baseline PPO `SARE` on seed `23`: greedy success `0.0000`, route entropy `1.3857`, active compute `0.5000`
- KL learner-state `SARE` on seed `23`: greedy success `1.0000`, route entropy `1.3804`, active compute `0.5000`
- published improved seed `19` remains in the same routing regime: route entropy `1.3837`, active compute `0.5000`
- bounded causal probes now cover seeds `7`, `19`, `23`, and `29`:
  - every single-expert ablation still drops seeds `7`, `19`, and `23` to `0.0000`
  - fixed-router override still drops all four probed seeds to `0.0000`
  - route randomization still drops seeds `7`, `19`, and `23` to `0.0000`, but only weakly harms seed `29`

So the strengthened DoorKey result is not just statistically non-collapsed routing. Under the bounded probe family used here, recovered performance is causally routing-dependent, though not every perturbation is equally strong on every recovered seed.

## Transfer Check

The exact same method does not transfer under the bounded KeyCorridor check.

Source artifact: [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md)

All three KeyCorridor seeds stayed flat:

- recovered `token_dense`: `0.0000`, `0.0000`, `0.0000`
- baseline PPO `SARE`: `0.0000`, `0.0000`, `0.0000`
- KL learner-state `SARE`: `0.0000`, `0.0000`, `0.0000`

## Recommendation

- Continue within DoorKey only.
- The DoorKey routed claim is now broader than it was in the claim-consolidation phase because it survives:
  - the missing matched `single_expert` fairness control on the original lane
  - one more fresh matched seed block
  - a broader, though not perfectly uniform, causal routing-dependence check
- Keep the scope explicit:
  - teacher-guided extraction only
  - DoorKey only
  - external `64`-episode evaluation only
- Do not broaden this into a PPO-only or cross-task routed advantage claim while the bounded KeyCorridor transfer check remains flat.
