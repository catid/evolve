# Memory Actor-Hidden Low-Rank Probe

- hypothesis: the live actor-hidden FiLM path may be too diffuse; a low-rank bottleneck could preserve the greedy conversion while reducing unnecessary hidden-state perturbation
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Train Return | Train Success | Low Rank | Rank Dim | Gate Mean | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.5172` | `0.5303` | `0.0` | `0` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `0.6509` | `0.6625` | `0.0` | `32` | `0.1433` | `0.1073` | `0.0228` |
| `por_actor_hidden_partial_shift22_lowrank16` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.2762` | `0.3077` | `1.0` | `16` | `0.1177` | `0.0068` | `0.0015` |
| `por_actor_hidden_partial_shift22_lowrank32` | `0.0000` | `0.0000` | `0.0000` | `0.0781` | `0.3706` | `0.3846` | `1.0` | `32` | `0.1344` | `0.0137` | `0.0023` |

## Deltas vs Partial-Shift22

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift22_lowrank16` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift22_lowrank32` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.3906` |

## Outcome

The low-rank variants did not produce a clear improvement over `partial_shift22`. If greedy conversion collapses or the lower/gap band does not improve, the live actor-hidden FiLM point still prefers the current full-width hidden update on this local surface.
