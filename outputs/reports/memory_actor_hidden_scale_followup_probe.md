# Memory Actor-Hidden Scale Follow-Up Probe

- hypothesis: the new `partial_shift225` winner may want slightly more hidden-FiLM scale than `0.35`; a moderate scale increase could keep greedy conversion while improving the lower or gap band
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Shift Weight | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift225` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6272` | `2.6143` | `2.5712` | `0.4279` | `0.4643` | `0.3340` | `0.1169` | `0.225` | `0.0987` | `0.0224` |
| `por_actor_hidden_partial_shift225_scale375` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.8639` | `2.8639` | `2.8639` | `0.4652` | `0.4730` | `0.3423` | `0.1284` | `0.225` | `0.1308` | `0.0287` |
| `por_actor_hidden_partial_shift225_scale40` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.5533` | `1.5533` | `1.5533` | `0.2504` | `0.3333` | `0.3736` | `0.1494` | `0.225` | `0.0986` | `0.0241` |

## Deltas vs Partial-Shift225 Reference

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift225_scale375` | `+0.0000` | `+0.0000` | `+0.0000` | `+0.0000` |
| `por_actor_hidden_partial_shift225_scale40` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |

## Interpretation

- This is a same-seed scale follow-up around the current `partial_shift225` best-known point.
- A scale-up variant only counts as interesting if it preserves the greedy conversion and improves the lower, gap, or shoulder band versus `partial_shift225`.
