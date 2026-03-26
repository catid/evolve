# Memory Actor-Hidden Scale Follow-Up Confirmation

- goal: resolve the 64-episode tie between `partial_shift225` and `partial_shift225_scale375` with a longer 256-episode confirmation pass
- task: `MiniGrid-MemoryS9-v0`
- confirmation episodes per mode: `256`
- checkpoints: fresh runs from `memory_actor_hidden_scale_followup_probe` plus the prior `partial_shift25` anchor

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Shift Weight | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3711` | `0.4414` | `0.4805` | `0.2990` | `0.4303` | `0.6881` | `0.5172` | `0.5303` | `0.000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift225` | `0.4688` | `0.4727` | `0.4727` | `0.4688` | `2.6143` | `2.5947` | `2.5555` | `0.4279` | `0.4643` | `0.225` | `0.0987` | `0.0224` |
| `por_actor_hidden_partial_shift225_scale375` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.8566` | `2.8566` | `2.8566` | `0.4652` | `0.4730` | `0.225` | `0.1308` | `0.0287` |
| `por_actor_hidden_partial_shift25` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6452` | `2.6452` | `2.6452` | `0.4940` | `0.5067` | `0.250` | `0.1230` | `0.0315` |

## Deltas vs Partial-Shift225 Reference

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift225_scale375` | `+0.0000` | `-0.0039` | `-0.0039` | `+0.0000` |
| `por_actor_hidden_partial_shift25` | `+0.0000` | `-0.0039` | `-0.0039` | `+0.0000` |

## Interpretation

- This is a 256-episode confirmation pass to resolve the live same-surface scale tie around the new `partial_shift225` point.
- A scale-up variant only counts as interesting if it keeps greedy conversion and creates a measurable lower/gap/shoulder advantage over `partial_shift225`.
