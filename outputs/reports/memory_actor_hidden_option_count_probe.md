# Memory Actor-Hidden Option-Count Probe

- hypothesis: the live `partial_shift22` Memory branch may be bottlenecked by too few POR options; a richer option bank could improve greedy or sampled decode while preserving the actor-hidden FiLM surface
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Options | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Train Return | Train Success | Throughput | Option Duration | Switch Rate | Gate Mean | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `4` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.5172` | `0.5303` | `116.9` | `2.9213` | `0.0049` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `4` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `0.6509` | `0.6625` | `92.6` | `1.6193` | `0.5179` | `0.1433` | `0.1073` | `0.0228` |
| `por_actor_hidden_partial_shift22_option6` | `6` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.3108` | `0.3846` | `91.1` | `1.1860` | `0.0000` | `0.1075` | `0.0420` | `0.0091` |
| `por_actor_hidden_partial_shift22_option8` | `8` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.4756` | `0.5000` | `92.1` | `1.7906` | `0.0025` | `0.1568` | `0.0314` | `0.0045` |

## Deltas vs Partial-Shift22

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift22_option6` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift22_option8` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |

## Outcome

Increasing POR option count on the `partial_shift22` surface did not produce a better point than the incumbent. Any changed option behavior failed to translate into a greedy-preserving improvement.
