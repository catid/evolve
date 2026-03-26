# Memory Actor-Hidden Partial-Shift Scale Probe

- hypothesis: the winning partial-shift branch may require the 0.25 shift but not the full 0.35 hidden-FiLM scale; a lower scale may preserve greedy conversion while reducing the small shoulder tradeoff and the extreme margin saturation
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Shift Weight | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.00` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift25_scale30` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.2360` | `1.2360` | `1.2360` | `0.4996` | `0.5357` | `0.3860` | `0.1158` | `0.25` | `0.0305` | `0.0072` |
| `por_actor_hidden_partial_shift25_scale325` | `0.0000` | `0.0000` | `0.0000` | `0.2656` | `0.3480` | `0.3480` | `0.3476` | `0.2817` | `0.5000` | `0.9665` | `0.3141` | `0.25` | `0.2266` | `0.0681` |
| `por_actor_hidden_partial_shift25` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6550` | `2.6550` | `2.6550` | `0.4940` | `0.5067` | `0.3544` | `0.1240` | `0.25` | `0.1230` | `0.0315` |

## Deltas vs Partial-Shift25 Reference

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift25_scale30` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift25_scale325` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.2031` |

## Interpretation

- This is a same-seed scale sweep around the new `partial_shift25` architecture win.
- A lower-scale variant only counts as interesting if it preserves the new greedy lift while improving the shoulder or reducing the lower/gap tradeoff versus the current `0.35` scale.
