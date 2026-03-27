# Memory Actor-Hidden Termination-Bias Probe

- hypothesis: the live `partial_shift22` branch may be slightly too switchy; a mild `termination_bias` change could improve persistence balance without destroying the greedy conversion
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Term Bias | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Option Duration | Switch Rate | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Throughput | Gate Signal | Gate Mean | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `1.00` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.0000` | `0.0000` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `117.04` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `1.00` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `0.0000` | `0.0000` | `2.3544` | `2.3544` | `2.3544` | `0.6509` | `0.6625` | `92.20` | `0.4094` | `0.1433` | `0.1073` | `0.0228` |
| `por_actor_hidden_partial_shift22_termbias075` | `0.75` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.9021` | `0.9021` | `0.9021` | `0.3939` | `0.4545` | `91.89` | `0.6740` | `0.2359` | `0.0730` | `0.0164` |
| `por_actor_hidden_partial_shift22_termbias125` | `1.25` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.7502` | `0.7502` | `0.7502` | `0.4219` | `0.4583` | `92.50` | `0.3076` | `0.1077` | `0.0224` | `0.0036` |

## Deltas vs Partial-Shift22

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder | Delta Duration | Delta Switch Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift22_termbias075` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` | `+0.0000` | `+0.0000` |
| `por_actor_hidden_partial_shift22_termbias125` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` | `+0.0000` | `+0.0000` |

- conclusion: mild `termination_bias` shifts do not improve on `partial_shift22`; this local greedy-conversion surface does not want a simple persistence-bias tweak
