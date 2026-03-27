# Memory Actor-Hidden Micro-Shift Probe

- hypothesis: the live plateau around `partial_shift22` may still have a finer optimum that keeps greedy conversion while slightly improving lower or gap behavior
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Shift Weight | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `1.3784` | `1.3329` | `1.3326` | `0.6431` | `0.6667` | `0.5822` | `0.2038` | `0.2200` | `0.3372` | `0.0482` |
| `por_actor_hidden_partial_shift221` | `0.0000` | `0.5156` | `0.4688` | `0.4844` | `0.2409` | `0.5726` | `0.8770` | `0.5314` | `0.5588` | `0.3347` | `0.1171` | `0.2210` | `0.0539` | `0.0106` |
| `por_actor_hidden_partial_shift2225` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.5833` | `0.5833` | `0.5833` | `0.5328` | `0.5500` | `0.3783` | `0.1324` | `0.2225` | `0.0728` | `0.0160` |

## Deltas vs Partial-Shift22

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift221` | `-0.4688` | `+0.0469` | `+0.0000` | `+0.0156` |
| `por_actor_hidden_partial_shift2225` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |

- conclusion: the current live plateau is already tight around `partial_shift22`; neither micro-shift produces a strictly better local architecture point on this fresh-root screen
