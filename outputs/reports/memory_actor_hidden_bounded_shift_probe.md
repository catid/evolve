# Memory Actor-Hidden Bounded-Shift Probe

- hypothesis: the live hidden-FiLM shift path is too brittle because the additive shift is unbounded; a tanh-bounded shift may preserve greedy conversion while stabilizing the sampled band
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Bound Shift | Bound Scale | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.0` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift22` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `1.3784` | `1.3329` | `1.3326` | `0.6431` | `0.6667` | `0.5822` | `0.2038` | `0.0` | `1.0000` | `0.3372` | `0.0482` |
| `por_actor_hidden_partial_shift22_bounded_shift` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.7747` | `1.7747` | `1.7747` | `0.2348` | `0.2800` | `0.3464` | `0.1212` | `1.0` | `1.0000` | `0.0317` | `0.0063` |
| `por_actor_hidden_partial_shift22_bounded_shift2x` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `2.2058` | `2.2058` | `2.2058` | `0.3689` | `0.3846` | `0.3019` | `0.1057` | `1.0` | `2.0000` | `0.0418` | `0.0249` |

## Deltas vs Partial-Shift22

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift22_bounded_shift` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift22_bounded_shift2x` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |

## Outcome

The bounded-shift variants did not produce a clear improvement over `partial_shift22`. If greedy conversion collapses or sampled lower/gap does not improve, the live actor-hidden FiLM point still prefers the current unbounded shift path on this local surface.
