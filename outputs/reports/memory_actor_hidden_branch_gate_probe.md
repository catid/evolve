# Memory Actor-Hidden Branch-Gate Probe

- hypothesis: the current actor-hidden FiLM winner may need branch-specific gating rather than a full-path gate mix; keep scale on raw duration and mix only the shift branch toward stability
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Train Return | Train Success | Branch Gates | Scale Mix | Shift Mix | Scale Gate | Shift Gate | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.5172` | `0.5303` | `0.0` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `0.6509` | `0.6625` | `0.0` | `1.0000` | `1.0000` | `0.1433` | `0.1433` | `0.1073` | `0.0228` |
| `por_actor_hidden_partial_shift22_shiftmix95` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.1893` | `0.2000` | `1.0` | `1.0000` | `0.9500` | `0.1294` | `0.1236` | `0.0266` | `0.0047` |
| `por_actor_hidden_partial_shift22_shiftmix90` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `0.4932` | `0.5000` | `1.0` | `1.0000` | `0.9000` | `0.1259` | `0.1142` | `0.0899` | `0.0149` |

## Deltas vs Partial-Shift22

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift22_shiftmix95` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift22_shiftmix90` | `+0.0000` | `+0.0000` | `+0.0000` | `+0.0000` |

## Outcome

The branch-specific shift-gate variants did not produce a clear improvement over `partial_shift22`. If greedy conversion collapses or the lower/gap band does not improve, the live actor-hidden FiLM point still prefers the raw-duration gate on both branches for this local surface.
