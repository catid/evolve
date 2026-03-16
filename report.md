# DoorKey Greedy-Recovery Report

## Current Conclusion

- `flat_dense` remains the strongest verified greedy DoorKey control.
- `token_dense` with `ppo.ent_coef=0.001` remains the canonical recovered tokenized DoorKey control.
- `single_expert` and `SARE` remain sampled-competent but greedy-failing on DoorKey.
- The bounded architecture-neutral recovery campaign is now complete:
  - checkpoint selection did not reveal any missed good greedy checkpoint
  - entropy schedules did not recover greedy `single_expert` or greedy `SARE`
  - self-imitation from successful sampled trajectories did not recover greedy `single_expert` or greedy `SARE`
- Current repo recommendation: stop routed work for greedy-policy claims on DoorKey unless a new, separately justified extraction method is proposed.

## Baseline And Reproduction

Source artifacts:

- [greedy_recovery_reproduction_note.md](outputs/reports/greedy_recovery_reproduction_note.md)
- [outputs/experiments/sare_retest/report.md](outputs/experiments/sare_retest/report.md)

Current reproduced DoorKey baseline:

| Variant | Greedy Success | Best Sampled Success | Train Return |
| --- | ---: | ---: | ---: |
| `flat_dense` | `1.000` | `1.000` | `0.960` |
| recovered `token_dense` | `0.750` | `1.000` | `0.942` |
| `single_expert` | `0.000` | `0.750` | `0.291` |
| `SARE` | `0.000` | `1.000` | `0.744` |

This is the accepted starting point for the greedy-recovery phase.

## Why The Gap Exists

Source artifacts:

- [policy_extraction_report.md](outputs/reports/policy_extraction_report.md)
- [tokenization_gap_report.md](outputs/reports/tokenization_gap_report.md)

The control decomposition is unchanged:

- `flat_dense` learns a sharp greedy policy.
- original `token_dense` was genuinely weak under partial observation, not just badly extracted.
- `single_expert` and `SARE` already contain strong sampled policies, but their greedy action ordering stays wrong.

That is why this phase focused on checkpoint choice, entropy schedules, and self-imitation rather than new routed architectures.

## Checkpoint Dynamics

Source artifact: [checkpoint_dynamics_report.md](outputs/reports/checkpoint_dynamics_report.md)

Result:

- `SARE` never shows a nonzero greedy checkpoint anywhere in the archived DoorKey series.
- `single_expert` never shows a nonzero greedy checkpoint anywhere in the archived DoorKey series.
- sampled-good checkpoints and greedy-good checkpoints do not merely diverge; the greedy-good checkpoints do not appear at all in the tested training traces.

This rules out “we just picked the wrong checkpoint” as the main explanation.

## Entropy Schedules

Source artifact: [entropy_schedule_report.md](outputs/reports/entropy_schedule_report.md)

Best bounded schedules:

| Variant | Best Schedule | Greedy Success | Best Sampled Success |
| --- | --- | ---: | ---: |
| `single_expert` | `late_linear:0.01->0.001@0.75` | `0.000` | `1.000` |
| `SARE` | `late_linear:0.01->0.001@0.75` | `0.000` | `1.000` |

Interpretation:

- simple entropy schedules can preserve or improve sampled competence
- they do not recover a usable greedy DoorKey policy for either `single_expert` or `SARE`
- large action margins alone are not enough; some schedules produce sharper wrong argmax decisions

So greedy recovery is not mainly an entropy-schedule problem in this repo.

## Self-Imitation

Source artifact: [self_imitation_report.md](outputs/reports/self_imitation_report.md)

Tested matrix:

- teacher mode: sampled `t=1.0`
- targets: `policy_head`, `policy_head_plus_last_shared`
- weightings: `uniform`, `return`

Result:

- every `SARE` self-imitation run stayed at greedy success `0.000`
- every `single_expert` self-imitation run stayed at greedy success `0.000`
- sampled competence was mostly preserved, but greedy extraction did not improve

This rules out the simplest successful-trajectory distillation path as a practical greedy recovery method for the current routed policies.

## Stop Condition

The campaign hit the memo’s stop condition:

- no bounded architecture-neutral sharpening family materially improved greedy `SARE`

Because checkpoint selection, entropy schedules, and self-imitation all failed to recover greedy `SARE`, the optional margin-regularization probe and KeyCorridor transfer check were not run. That would have expanded scope after the repo had already reached a clean no-go answer for the current DoorKey question.

## Recommendation

- Keep `flat_dense` as the strongest verified greedy DoorKey control.
- Keep `token_dense` with `ppo.ent_coef=0.001` as the canonical recovered tokenized control.
- Treat current `single_expert` and `SARE` DoorKey policies as sampled-competent but not greedily recoverable under the bounded, architecture-neutral methods tested here.
- Pause or stop routed work for greedy-policy claims in this repo.
- If work continues at all, it should be framed as a new extraction-method project, not as evidence that current routed models are close to a greedy win.
