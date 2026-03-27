# Memory Actor-Hidden Split-Heads Probe

- hypothesis: the live actor-hidden FiLM surface may be limited by a shared head that entangles multiplicative and additive adjustments; separate scale and shift heads may preserve greedy conversion while recovering some lower-band lift
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Split Heads | Gate Signal | Gate Mean | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `1.3784` | `1.3329` | `1.3326` | `0.6431` | `0.6667` | `0.0` | `0.5822` | `0.2038` | `0.3372` | `0.0482` |
| `por_actor_hidden_partial_shift22_split_heads` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.7915` | `1.7915` | `1.7915` | `0.4228` | `0.4667` | `1.0` | `0.3317` | `0.1161` | `0.0450` | `0.0075` |
| `por_actor_hidden_partial_shift221_split_heads` | `0.0000` | `0.0000` | `0.0000` | `0.0312` | `0.7254` | `0.7254` | `0.7256` | `0.4209` | `0.4419` | `1.0` | `0.4590` | `0.1607` | `0.0798` | `0.0171` |

## Deltas vs Partial-Shift22

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift22_split_heads` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift221_split_heads` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4375` |

## Outcome

The split-head variants did not produce a clear improvement over `partial_shift22`. If greedy conversion collapses or the lower/gap band does not improve, the live actor-hidden FiLM point still prefers the original shared scale/shift head on this local surface.
