# Memory Actor-Hidden Shift22 Scale-Up Probe

- hypothesis: the new `partial_shift22` point may want a very small hidden-FiLM scale increase above `0.35`, keeping greedy conversion while improving lower or gap behavior
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Shift Weight | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `1.3784` | `1.3329` | `1.3326` | `0.6431` | `0.6667` | `0.5822` | `0.2038` | `0.2200` | `0.3372` | `0.0482` |
| `por_actor_hidden_partial_shift22_scale355` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.4662` | `1.4662` | `1.4662` | `0.4562` | `0.4762` | `0.4966` | `0.1763` | `0.2200` | `0.0939` | `0.0233` |
| `por_actor_hidden_partial_shift22_scale36` | `0.0000` | `0.0000` | `0.0000` | `0.0156` | `0.7848` | `0.7848` | `0.7834` | `0.6238` | `0.6364` | `0.3608` | `0.1299` | `0.2200` | `0.0376` | `0.0066` |

## Deltas vs Partial-Shift22

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift22_scale355` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift22_scale36` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4531` |

- conclusion: raising scale above `0.35` around `partial_shift22` does not produce a strictly better local architecture point on this fresh-root screen
