# Memory Actor-Hidden Gate-Bias Probe

- hypothesis: `partial_shift22` may be limited by a fixed duration-gate threshold; a learned gate-bias head could preserve greedy conversion while recovering some sampled-band headroom by shifting that gate on a per-state basis
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Bias | Bias Scale | Bias Mean | Gate Mean | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0` | `0.00` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.3544` | `2.3544` | `2.3544` | `0.6509` | `0.6625` | `0.4094` | `0` | `0.00` | `0.0000` | `0.1433` | `0.1073` | `0.0228` |
| `por_actor_hidden_partial_shift22_gatebias10` | `0.0000` | `0.4844` | `0.4375` | `0.4844` | `0.1181` | `0.7053` | `0.8830` | `0.6074` | `0.6364` | `0.2823` | `1` | `0.10` | `0.0055` | `0.1007` | `0.0718` | `0.0145` |
| `por_actor_hidden_partial_shift22_gatebias20` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.5104` | `1.5104` | `1.5104` | `0.4125` | `0.5000` | `0.3224` | `1` | `0.20` | `-0.0207` | `0.1056` | `0.0373` | `0.0072` |

## Deltas vs Partial-Shift22

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift22_gatebias10` | `-0.4688` | `+0.0156` | `-0.0312` | `+0.0156` |
| `por_actor_hidden_partial_shift22_gatebias20` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |

- conclusion: the gate-bias variants do not improve on `partial_shift22`; this local actor-hidden surface does not benefit from a learned shared threshold shift on the duration gate
