# Memory Actor-Hidden Mid-Shift Confirmation

- goal: break the exact 64-episode tie between `partial_shift225`, `partial_shift24`, and the incumbent `partial_shift25` using a longer 256-episode confirmation pass
- task: `MiniGrid-MemoryS9-v0`
- confirmation episodes per mode: `256`
- checkpoints: fresh runs from `memory_actor_hidden_mid_shift_probe`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Shift Weight | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3711` | `0.4414` | `0.4805` | `0.2990` | `0.4303` | `0.6881` | `0.5172` | `0.5303` | `0.000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift225` | `0.4688` | `0.4727` | `0.4727` | `0.4688` | `2.6143` | `2.5947` | `2.5555` | `0.4279` | `0.4643` | `0.225` | `0.0987` | `0.0224` |
| `por_actor_hidden_partial_shift24` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.9897` | `2.9897` | `2.9897` | `0.5287` | `0.5362` | `0.240` | `0.1548` | `0.0389` |
| `por_actor_hidden_partial_shift25` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6452` | `2.6452` | `2.6452` | `0.4940` | `0.5067` | `0.250` | `0.1230` | `0.0315` |

## Deltas vs Partial-Shift25 Reference

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift225` | `+0.0000` | `+0.0039` | `+0.0039` | `+0.0000` |
| `por_actor_hidden_partial_shift24` | `+0.0000` | `+0.0000` | `+0.0000` | `+0.0000` |

## Interpretation

- This is a 256-episode confirmation pass to resolve the 64-episode tie inside the live mid-band threshold region.
- A new midpoint only counts as interesting if it keeps the greedy conversion and creates a measurable lower/gap/shoulder advantage over `partial_shift25`.
