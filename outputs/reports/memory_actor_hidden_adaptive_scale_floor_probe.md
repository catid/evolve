# Memory Actor-Hidden Adaptive Scale-Floor Probe

- hypothesis: `partial_shift22` needs the full `0.35` scale on high-duration states for greedy conversion, but sampled behavior may improve if lower-duration states interpolate toward a softer scale floor instead of always using the full scale
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Adaptive Floor | Floor Value | Adaptive Scale Mean | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0` | `0.00` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.3544` | `2.3544` | `2.3544` | `0.6509` | `0.6625` | `0.4094` | `0.1433` | `0` | `0.00` | `0.1433` | `0.1073` | `0.0228` |
| `por_actor_hidden_partial_shift22_adaptive_floor25` | `0.3125` | `0.4688` | `0.4844` | `0.4688` | `1.0770` | `1.4543` | `1.5349` | `0.4404` | `0.4500` | `0.3941` | `0.1260` | `1` | `0.25` | `0.1141` | `0.0285` | `0.0078` |
| `por_actor_hidden_partial_shift22_adaptive_floor30` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.3224` | `1.3224` | `1.3224` | `0.7194` | `0.7500` | `0.3459` | `0.1154` | `1` | `0.30` | `0.1098` | `0.0604` | `0.0157` |

## Deltas vs Partial-Shift22

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift22_adaptive_floor25` | `-0.1562` | `+0.0000` | `+0.0156` | `+0.0000` |
| `por_actor_hidden_partial_shift22_adaptive_floor30` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |

- conclusion: at least one adaptive scale-floor variant preserves nonzero greedy and improves part of the sampled band; run a 256-episode confirmation pass before treating it as real
