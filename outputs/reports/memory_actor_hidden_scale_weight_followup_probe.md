# Memory Actor-Hidden Mild Scale-Weight Follow-Up Probe

- hypothesis: the current `partial_shift225` best point may want a slightly weaker multiplicative scale branch while keeping the same additive shift; a mild split-scale reduction could preserve greedy conversion while improving the sampled band
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Scale Weight | Shift Weight | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `1.00` | `0.000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift225` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6272` | `2.6143` | `2.5712` | `0.4279` | `0.4643` | `0.3340` | `0.1169` | `1.00` | `0.225` | `0.0987` | `0.0224` |
| `por_actor_hidden_partial_shift225_scaleweight90` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.3068` | `0.3068` | `0.3068` | `0.3801` | `0.4286` | `0.2986` | `0.1045` | `0.90` | `0.225` | `0.0247` | `0.0029` |
| `por_actor_hidden_partial_shift225_scaleweight95` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.8858` | `1.8858` | `1.8858` | `0.1390` | `0.1429` | `0.3401` | `0.1190` | `0.95` | `0.225` | `0.0567` | `0.0114` |

## Deltas vs Partial-Shift225 Reference

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift225_scaleweight90` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift225_scaleweight95` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |

## Interpretation

- This is a same-seed mild split scale-weight follow-up around the current `partial_shift225` best-known point.
- A mild split scale-weight variant only counts as interesting if it preserves the greedy conversion and improves the lower, gap, or shoulder band versus `partial_shift225`.
