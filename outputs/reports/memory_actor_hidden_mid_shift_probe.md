# Memory Actor-Hidden Mid-Shift Probe

- hypothesis: the greedy-conversion onset may sit just below the current `partial_shift25` winner; a slightly smaller hidden-FiLM shift could preserve the greedy lift while improving the lower or gap band before the onset disappears
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Shift Weight | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift225` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6272` | `2.6143` | `2.5712` | `0.4279` | `0.4643` | `0.3340` | `0.1169` | `0.225` | `0.0987` | `0.0224` |
| `por_actor_hidden_partial_shift24` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `3.0006` | `3.0006` | `3.0006` | `0.5287` | `0.5362` | `0.5870` | `0.2055` | `0.240` | `0.1548` | `0.0389` |
| `por_actor_hidden_partial_shift25` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6550` | `2.6550` | `2.6550` | `0.4940` | `0.5067` | `0.3544` | `0.1240` | `0.250` | `0.1230` | `0.0315` |

## Deltas vs Base

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift225` | `+0.4688` | `+0.0938` | `+0.0156` | `+0.0000` |
| `por_actor_hidden_partial_shift24` | `+0.4688` | `+0.0938` | `+0.0156` | `+0.0000` |
| `por_actor_hidden_partial_shift25` | `+0.4688` | `+0.0938` | `+0.0156` | `+0.0000` |

## Deltas vs Partial-Shift25 Reference

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift225` | `+0.0000` | `+0.0000` | `+0.0000` | `+0.0000` |
| `por_actor_hidden_partial_shift24` | `+0.0000` | `+0.0000` | `+0.0000` | `+0.0000` |

## Interpretation

- This is a same-seed mid-band threshold sweep around the current `partial_shift25` architecture win.
- A mid-band variant only counts as interesting if it preserves the greedy lift while improving the lower or gap band versus `partial_shift25`.
