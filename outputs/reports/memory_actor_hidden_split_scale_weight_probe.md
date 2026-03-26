# Memory Actor-Hidden Split Scale-Weight Probe

- hypothesis: the actor-hidden partial_shift25 win may require the current shift branch but not the full multiplicative scale branch; reducing only the scale path could preserve greedy conversion while softening the saturation and shoulder tradeoff
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Scale Weight | Shift Weight | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `1.00` | `0.00` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift25_scaleweight50` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.4027` | `0.4027` | `0.4026` | `0.0000` | `0.0000` | `0.3370` | `0.1179` | `0.50` | `0.25` | `0.0177` | `0.0119` |
| `por_actor_hidden_partial_shift25_scaleweight75` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.8301` | `0.8301` | `0.8301` | `0.3185` | `0.3333` | `0.3756` | `0.1315` | `0.75` | `0.25` | `0.0307` | `0.0099` |
| `por_actor_hidden_partial_shift25` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6550` | `2.6550` | `2.6550` | `0.4940` | `0.5067` | `0.3544` | `0.1240` | `1.00` | `0.25` | `0.1230` | `0.0315` |

## Deltas vs Partial-Shift25 Reference

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift25_scaleweight50` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift25_scaleweight75` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |

## Interpretation

- This is a same-seed split scale-weight probe around the current `partial_shift25` architecture win.
- A split scale-weight variant only counts as interesting if it preserves the new greedy lift while improving the sampled band or softening the shoulder tradeoff versus the incumbent.
