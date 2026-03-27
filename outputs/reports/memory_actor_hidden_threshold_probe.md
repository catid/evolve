# Memory Actor-Hidden Threshold Probe

- hypothesis: the greedy-conversion threshold may sit between sampled-only `partial_shift20` and the current `partial_shift225` winner; a narrower midpoint could preserve greedy while recovering some sampled-band lift
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Shift Weight | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift225` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6272` | `2.6143` | `2.5712` | `0.4279` | `0.4643` | `0.3340` | `0.1169` | `0.225` | `0.0987` | `0.0224` |
| `por_actor_hidden_partial_shift215` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.9840` | `1.9840` | `1.9840` | `0.5469` | `0.5750` | `0.5129` | `0.1795` | `0.215` | `0.0957` | `0.0227` |
| `por_actor_hidden_partial_shift22` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `1.3784` | `1.3329` | `1.3326` | `0.6431` | `0.6667` | `0.5822` | `0.2038` | `0.220` | `0.3372` | `0.0482` |

## Deltas vs Partial-Shift225

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift215` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift22` | `+0.0000` | `+0.0000` | `+0.0000` | `+0.0000` |

- conclusion: at least one threshold midpoint survives or improves on part of the current surface and needs a 256-episode confirmation pass
