# Memory Actor-Hidden Lower-Shift Probe

- hypothesis: the new actor-hidden partial-shift win may sit above the minimum shift needed for greedy conversion; a smaller shift might preserve the greedy lift while softening the slight shoulder regression and reducing margin saturation
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Shift Weight | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.00` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift15` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0527` | `0.0527` | `0.0528` | `0.4609` | `0.5500` | `0.3521` | `0.1232` | `0.15` | `0.0435` | `0.0046` |
| `por_actor_hidden_partial_shift20` | `0.0000` | `0.5000` | `0.4688` | `0.4375` | `0.2253` | `0.1679` | `0.1658` | `0.2333` | `0.5000` | `0.4198` | `0.1469` | `0.20` | `0.0502` | `0.0076` |
| `por_actor_hidden_partial_shift25` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6550` | `2.6550` | `2.6550` | `0.4940` | `0.5067` | `0.3544` | `0.1240` | `0.25` | `0.1230` | `0.0315` |

## Deltas vs Base

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift15` | `+0.0000` | `-0.3750` | `-0.4531` | `-0.4688` |
| `por_actor_hidden_partial_shift20` | `+0.0000` | `+0.1250` | `+0.0156` | `-0.0312` |
| `por_actor_hidden_partial_shift25` | `+0.4688` | `+0.0938` | `+0.0156` | `+0.0000` |

## Deltas vs Partial-Shift25 Reference

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift15` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift20` | `-0.4688` | `+0.0312` | `+0.0000` | `-0.0312` |

## Interpretation

- This is a same-seed lower-shift follow-up around the new `partial_shift25` win.
- A lower-shift variant only counts as interesting if it preserves the new greedy lift and improves the shoulder or lowers the lower/gap tradeoff versus `partial_shift25`.
