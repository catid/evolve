# Memory Actor-Hidden Scale-Branch Gate Probe

- hypothesis: the current actor-hidden FiLM winner may want a branch-specific gate on the multiplicative scale path only; keep the successful raw-duration shift branch intact and mix only scale toward stability
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Train Return | Train Success | Branch Gates | Scale Mix | Shift Mix | Scale Gate | Shift Gate | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.5172` | `0.5303` | `0.0` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `0.6509` | `0.6625` | `0.0` | `1.0000` | `1.0000` | `0.1433` | `0.1433` | `0.1073` | `0.0228` |
| `por_actor_hidden_partial_shift22_scalemix95` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.5339` | `0.6250` | `1.0` | `0.9500` | `1.0000` | `0.1322` | `0.1387` | `0.0311` | `0.0061` |
| `por_actor_hidden_partial_shift22_scalemix90` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.2694` | `0.2778` | `1.0` | `0.9000` | `1.0000` | `0.1620` | `0.1708` | `0.1078` | `0.0272` |

## Deltas vs Partial-Shift22

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift22_scalemix95` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift22_scalemix90` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |

## Outcome

The scale-branch-only gate mixes did not produce a clear improvement over `partial_shift22`. If greedy conversion collapses or the sampled band does not improve, the live actor-hidden FiLM point still prefers the raw-duration gate on the scale branch for this local surface.
