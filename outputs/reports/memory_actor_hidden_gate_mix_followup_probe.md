# Memory Actor-Hidden Mild Gate-Mix Follow-Up Probe

- hypothesis: the current `partial_shift225` best point may want a slight duration-to-stability gate mix while keeping the same hidden-FiLM shift and scale; a mild mix could preserve greedy conversion while smoothing the sampled band
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Duration Mix | Shift Weight | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `1.00` | `0.000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift225` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6272` | `2.6143` | `2.5712` | `0.4279` | `0.4643` | `0.3340` | `0.1169` | `1.00` | `0.225` | `0.0987` | `0.0224` |
| `por_actor_hidden_partial_shift225_mix90` | `0.0000` | `0.3438` | `0.3438` | `0.3438` | `0.1505` | `0.1322` | `0.1300` | `0.3722` | `0.5000` | `0.3061` | `0.1071` | `0.90` | `0.225` | `0.0293` | `0.0058` |
| `por_actor_hidden_partial_shift225_mix95` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.6635` | `1.6635` | `1.6635` | `0.5974` | `0.6364` | `0.3382` | `0.1184` | `0.95` | `0.225` | `0.0280` | `0.0061` |

## Deltas vs Partial-Shift225 Reference

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift225_mix90` | `-0.4688` | `-0.1250` | `-0.1250` | `-0.1250` |
| `por_actor_hidden_partial_shift225_mix95` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |

## Interpretation

- This is a same-seed mild gate-mix follow-up around the current `partial_shift225` best-known point.
- A mild gate-mix variant only counts as interesting if it preserves the greedy conversion and improves the lower, gap, or shoulder band versus `partial_shift225`.
