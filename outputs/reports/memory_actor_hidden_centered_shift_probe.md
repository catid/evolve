# Memory Actor-Hidden Centered Shift Probe

- hypothesis: the live `partial_shift22` actor-hidden FiLM branch may be carrying a harmful per-step channelwise DC offset, so mean-centering the additive shift could preserve the greedy conversion while improving lower-band decode quality
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Train Return | Train Success | Center Shift | Center Scale | Center Mean Abs | Gate Mean | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.5172` | `0.5303` | `0.0` | `0.00` | `0.000000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `0.6509` | `0.6625` | `0.0` | `1.00` | `0.000157` | `0.1433` | `0.0228` |
| `por_actor_hidden_partial_shift22_centered_shift` | `0.0000` | `0.0312` | `0.0469` | `0.4062` | `0.4156` | `0.4247` | `1.0` | `1.00` | `0.000000` | `0.1041` | `0.0091` |
| `por_actor_hidden_partial_shift22_centered_shift2x` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.5578` | `0.6000` | `1.0` | `2.00` | `0.000000` | `0.1098` | `0.0199` |

## Deltas vs Partial-Shift22

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift22_centered_shift` | `-0.4688` | `-0.4375` | `-0.4219` | `-0.0625` |
| `por_actor_hidden_partial_shift22_centered_shift2x` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |

## Outcome

Centering the actor-hidden additive shift did not produce a better point than `partial_shift22`. Any live behavior that remained failed to improve the lower/gap band while keeping the current greedy conversion.
