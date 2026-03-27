# Memory Actor-Hidden Shift22 Scale Probe

- hypothesis: the new `partial_shift22` point may tolerate a small hidden-FiLM scale trim below `0.35`, keeping greedy conversion while improving lower or gap behavior
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Shift Weight | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `1.3784` | `1.3329` | `1.3326` | `0.6431` | `0.6667` | `0.5822` | `0.2038` | `0.2200` | `0.3372` | `0.0482` |
| `por_actor_hidden_partial_shift22_scale345` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.1596` | `0.1596` | `0.1596` | `0.1872` | `0.3750` | `0.4254` | `0.1468` | `0.2200` | `0.0534` | `0.0104` |
| `por_actor_hidden_partial_shift22_scale34` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.6441` | `0.6441` | `0.6441` | `0.2215` | `0.3333` | `0.4313` | `0.1466` | `0.2200` | `0.0285` | `0.0049` |

## Deltas vs Partial-Shift22

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift22_scale345` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift22_scale34` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |

- conclusion: trimming scale below `0.35` around `partial_shift22` does not produce a strictly better local architecture point on this fresh-root screen
