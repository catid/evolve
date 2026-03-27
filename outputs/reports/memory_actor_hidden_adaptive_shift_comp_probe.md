# Memory Actor-Hidden Adaptive Shift-Compensation Probe

- hypothesis: the adaptive-floor branch loses too much greedy signal because the additive hidden shift shrinks alongside the softened scale branch; compensating the shift when scale is softened may recover greedy without giving up the sampled-band gain
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `5`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Adaptive Scale Mean | Shift Compensation Mean | Comp Enabled | Comp Scale | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.0000` | `0` | `0.00` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.3544` | `2.3544` | `2.3544` | `0.6509` | `0.6625` | `0.4094` | `0.1433` | `0.1433` | `0` | `0.00` | `0.1073` | `0.0228` |
| `por_actor_hidden_partial_shift22_adaptive_floor25` | `0.3125` | `0.4688` | `0.4844` | `0.4688` | `1.0770` | `1.4543` | `1.5349` | `0.4404` | `0.4500` | `0.3941` | `0.1141` | `0.1380` | `0` | `0.00` | `0.0285` | `0.0078` |
| `por_actor_hidden_partial_shift22_adaptive_floor25_shiftcomp50` | `0.0000` | `0.5312` | `0.5312` | `0.5469` | `0.0295` | `1.5246` | `1.5603` | `0.5378` | `0.5532` | `0.3640` | `0.1043` | `0.1390` | `1` | `0.50` | `0.0207` | `0.0041` |
| `por_actor_hidden_partial_shift22_adaptive_floor25_shiftcomp100` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.7746` | `1.7746` | `1.7746` | `0.4670` | `0.5833` | `0.3347` | `0.0949` | `0.1394` | `1` | `1.00` | `0.0377` | `0.0139` |

## Deltas vs Adaptive-Floor25

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift22_adaptive_floor25_shiftcomp50` | `-0.3125` | `+0.0625` | `+0.0469` | `+0.0781` |
| `por_actor_hidden_partial_shift22_adaptive_floor25_shiftcomp100` | `-0.3125` | `-0.4688` | `-0.4844` | `-0.4688` |

- conclusion: shift compensation does not improve on `adaptive_floor25`; the current adaptive-floor trade-off is not fixed by simply boosting the additive hidden shift
