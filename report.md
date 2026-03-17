# DoorKey Extraction Report

## Current Conclusion

- `flat_dense` remains the strongest greedy DoorKey control.
- recovered `token_dense` (`ppo.ent_coef=0.001`) remains the canonical tokenized control.
- PPO-only `SARE` still loses the fair greedy DoorKey comparison.
- The only positive routed result is still teacher-guided:
  - teacher-logit `KL` learner-state supervision for `SARE`
  - DoorKey only
  - external `64`-episode evaluation only
- After the final fairness-and-failure analysis pass, the best description of the result is:
  - a bounded DoorKey teacher-guided `SARE` win
  - narrower and more method-first than the earlier broadened DoorKey reading
  - not strong enough to promote into a specifically multi-expert routed DoorKey edge

## Final Decision Path

Source artifacts:

- [lss_frozen_claim_reproduction_note.md](outputs/reports/lss_frozen_claim_reproduction_note.md)
- [lss_final_block_single_expert_control_report.md](outputs/reports/lss_final_block_single_expert_control_report.md)
- [lss_final_block_failure_analysis.md](outputs/reports/lss_final_block_failure_analysis.md)
- [lss_frozen_claim_updated_combined_doorkey_report.md](outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.md)
- [lss_frozen_claim_decision_memo.md](outputs/reports/lss_frozen_claim_decision_memo.md)
- [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md)

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

So the repo’s positive routed result still does not come from PPO tuning or offline imitation.

## What Changed In The Frozen-Claim Analysis Phase

The final fairness-and-failure phase answered three remaining DoorKey-only questions:

- does `SARE` stay ahead once the missing final-block matched `single_expert` control is added?
- why does the final fresh block `47/53/59` flip toward `token_dense`?
- does the updated combined DoorKey fairness picture justify thawing the claim?

The answer is:

- no on the final-block `single_expert` fairness check
- the weak final-block seeds look more like extraction mismatch than bad teacher labels
- no on thawing the claim
- still no on bounded KeyCorridor transfer

On the matched structured DoorKey slice where all three structured students exist:

| Lane | Seed | recovered `token_dense` | KL learner-state `token_dense` | KL learner-state `single_expert` | baseline PPO `SARE` | KL learner-state `SARE` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `original` | `7` | `0.7031` | `1.0000` | `1.0000` | `0.0000` | `1.0000` |
| `original` | `11` | `0.0000` | `0.0000` | `1.0000` | `0.0000` | `0.5625` |
| `original` | `19` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `0.5781` |
| `fresh` | `23` | `0.0000` | `0.0000` | `0.4062` | `0.0000` | `1.0000` |
| `fresh` | `29` | `0.0000` | `0.6250` | `1.0000` | `0.0000` | `1.0000` |
| `fresh` | `31` | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `1.0000` |
| `fresh_extra` | `37` | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `1.0000` |
| `fresh_extra` | `41` | `0.0000` | `0.0000` | `0.4375` | `0.0000` | `1.0000` |
| `fresh_extra` | `43` | `0.0000` | `0.0000` | `1.0000` | `0.0000` | `0.4688` |

Mean greedy success on that structured slice:

- `KL` learner-state `token_dense`: `0.5139`
- `KL` learner-state `single_expert`: `0.7604`
- `KL` learner-state `SARE`: `0.8455`

On the final fresh matched DoorKey block:

| Seed | recovered `token_dense` | KL learner-state `token_dense` | KL learner-state `single_expert` | baseline PPO `SARE` | KL learner-state `SARE` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `47` | `1.0000` | `1.0000` | `0.4531` | `0.0000` | `0.0000` |
| `53` | `1.0000` | `1.0000` | `0.5156` | `0.0000` | `0.5156` |
| `59` | `1.0000` | `1.0000` | `0.4219` | `0.0000` | `0.4219` |

Mean greedy success on that block:

- recovered `token_dense`: `1.0000`
- `KL` learner-state `token_dense`: `1.0000`
- `KL` learner-state `single_expert`: `0.4635`
- `KL` learner-state `SARE`: `0.3125`

Across the final combined DoorKey picture:

| Variant | Mean Greedy Success | Complete-Seed Failures |
| --- | ---: | ---: |
| recovered `token_dense` | `0.5586` | `5` |
| `KL` learner-state `token_dense` | `0.6354` | `4` |
| `KL` learner-state `single_expert` | `0.6862` | `1` |
| baseline PPO `SARE` | `0.0000` | `12` |
| `KL` learner-state `SARE` | `0.7122` | `1` |

So the right claim after the frozen-claim analysis pass is:

- teacher-guided KL learner-state supervision helps structured students generally
- `SARE` still ends slightly ahead of the matched structured controls in the full combined DoorKey picture
- but the final fresh block plus the missing `single_expert` fairness control push the result back toward a method-first interpretation
- the evidence is no longer strong enough to strengthen the claim into a specifically multi-expert routed DoorKey edge

## Route Dependence

The recovered DoorKey `SARE` checkpoints still look routed, and the current phase adds broader causal evidence that performance depends on routing even though the overall claim stays bounded.

Source artifacts:

- [lss_new_case_route_integrity_report.md](outputs/reports/lss_new_case_route_integrity_report.md)
- [lss_causal_route_dependence_report.md](outputs/reports/lss_causal_route_dependence_report.md)
- [lss_seed29_route_randomization_forensics.md](outputs/reports/lss_seed29_route_randomization_forensics.md)
- [lss_broader_route_dependence_report.md](outputs/reports/lss_broader_route_dependence_report.md)

Key result:

- baseline PPO `SARE` on seed `23`: greedy success `0.0000`, route entropy `1.3857`, active compute `0.5000`
- KL learner-state `SARE` on seed `23`: greedy success `1.0000`, route entropy `1.3804`, active compute `0.5000`
- bounded causal probes now cover seeds `7`, `19`, `23`, `29`, `31`, and `37`
  - fixed-router override has mean greedy-success drop `0.9297`
  - worst expert ablation has mean greedy-success drop `0.9297`
  - current route-randomization has mean greedy-success drop `0.7031`
  - seed `29` is the main narrow exception: repeated current and stronger randomization ladders stay high there, even though fixed-pair overrides still collapse success

So the DoorKey result is not just statistically non-collapsed routing. Under the bounded probe family used here, recovered performance remains causally routing-dependent, but the mechanism claim is narrower than a clean “all route randomization kills performance” story.

## Transfer Check

The exact same method still does not transfer under the bounded KeyCorridor check.

Source artifact: [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md)

All three KeyCorridor seeds stayed flat:

- recovered `token_dense`: `0.0000`, `0.0000`, `0.0000`
- baseline PPO `SARE`: `0.0000`, `0.0000`, `0.0000`
- KL learner-state `SARE`: `0.0000`, `0.0000`, `0.0000`

## Recommendation

- Stay frozen at the current DoorKey-only scope, and narrow the wording further.
- The current evidence still supports a bounded teacher-guided DoorKey `SARE` win, but it is not strong enough to strengthen into a specifically multi-expert routed claim because:
  - matched `single_expert` beats or matches `SARE` on the final fresh block
  - the weak final-block seeds show extraction mismatch without a teacher-quality failure
  - one final fresh matched DoorKey block materially weakens the edge
- Keep the scope explicit:
  - teacher-guided extraction only
  - DoorKey only
  - external `64`-episode evaluation only
- Do not broaden this into a PPO-only, specifically multi-expert, or cross-task routed advantage claim while the bounded KeyCorridor transfer check remains flat.
