# Memory Actor-Hidden Shift221 Scale Probe

- hypothesis: the sampled-strong but greedy-dead `partial_shift221` branch may be under-scaled; small scale increases at the same shift could recover greedy without giving back the lower-band lift
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `5`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | FiLM Scale | Shift | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Train Return | Train Success | Gate Mean | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.500` | `1.0000` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.350` | `0.2200` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `0.6509` | `0.6625` | `0.1433` | `0.1073` | `0.0228` |
| `por_actor_hidden_partial_shift221` | `0.350` | `0.2210` | `0.0000` | `0.0469` | `0.0469` | `0.1250` | `0.3720` | `0.4000` | `0.1737` | `0.0325` | `0.0048` |
| `por_actor_hidden_partial_shift221_scale375` | `0.375` | `0.2210` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.4872` | `0.5128` | `0.1992` | `0.1133` | `0.0231` |
| `por_actor_hidden_partial_shift221_scale40` | `0.400` | `0.2210` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.5621` | `0.5814` | `0.1485` | `0.0559` | `0.0084` |

## Deltas vs References

| Variant | Delta Greedy vs Shift22 | Delta Lower vs Shift22 | Delta Gap vs Shift22 | Delta Shoulder vs Shift22 | Delta Greedy vs Shift221 | Delta Lower vs Shift221 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift221_scale375` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` | `+0.0000` | `-0.0469` |
| `por_actor_hidden_partial_shift221_scale40` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` | `+0.0000` | `-0.0469` |

## Outcome

Raising FiLM scale on the `partial_shift221` branch did not produce a better point than `partial_shift22`. Any surviving sampled behavior failed to recover greedy while preserving the lower-band advantage.
