# Memory Actor-Hidden Upper-Shift Probe

- hypothesis: the current `partial_shift25` winner may still be below the true optimum; a modestly higher hidden-FiLM shift could preserve the greedy conversion while improving the lower or gap band before the known collapse regime
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Shift Weight | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.00` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift25` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6550` | `2.6550` | `2.6550` | `0.4940` | `0.5067` | `0.3544` | `0.1240` | `0.25` | `0.1230` | `0.0315` |
| `por_actor_hidden_partial_shift30` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.7183` | `1.7183` | `1.7183` | `0.4331` | `0.5000` | `0.3345` | `0.1171` | `0.30` | `0.0249` | `0.0060` |
| `por_actor_hidden_partial_shift35` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.5954` | `0.5823` | `0.5288` | `0.3243` | `0.3500` | `0.3165` | `0.1108` | `0.35` | `0.0527` | `0.0195` |

## Deltas vs Base

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift25` | `+0.4688` | `+0.0938` | `+0.0156` | `+0.0000` |
| `por_actor_hidden_partial_shift30` | `+0.0000` | `-0.3750` | `-0.4531` | `-0.4688` |
| `por_actor_hidden_partial_shift35` | `+0.0000` | `-0.3750` | `-0.4531` | `-0.4688` |

## Deltas vs Partial-Shift25 Reference

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift30` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift35` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |

## Interpretation

- This is a same-seed upper-shift follow-up around the current `partial_shift25` architecture win.
- An upper-shift variant only counts as interesting if it preserves the greedy lift and improves the lower or gap band without collapsing the shoulder versus `partial_shift25`.
