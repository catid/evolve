# Memory Actor-Hidden Partial-Shift FiLM Probe

- hypothesis: the mild scale-only hidden FiLM branch shows the shift term is not required for gap/shoulder behavior, but a small shift may recover the remaining lower-band lift without reopening the full-film tradeoff
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Shift Weight | Scale Only | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.00` | `0` | `0.0000` | `0.0000` |
| `por_actor_hidden_scale_film_mild` | `0.0000` | `0.4219` | `0.4688` | `0.4531` | `0.2755` | `0.2915` | `0.3231` | `0.2737` | `0.3333` | `0.3312` | `0.1159` | `1.00` | `1` | `0.0314` | `0.0000` |
| `por_actor_hidden_partial_shift25` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6550` | `2.6550` | `2.6550` | `0.4940` | `0.5067` | `0.3544` | `0.1240` | `0.25` | `0` | `0.1230` | `0.0315` |
| `por_actor_hidden_partial_shift50` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.7004` | `1.7004` | `1.7004` | `0.4838` | `0.5000` | `0.3445` | `0.1206` | `0.50` | `0` | `0.0286` | `0.0207` |

## Deltas vs Base

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_scale_film_mild` | `+0.0000` | `+0.0469` | `+0.0156` | `-0.0156` |
| `por_actor_hidden_partial_shift25` | `+0.4688` | `+0.0938` | `+0.0156` | `+0.0000` |
| `por_actor_hidden_partial_shift50` | `+0.0000` | `-0.3750` | `-0.4531` | `-0.4688` |

## Deltas vs Mild Scale-Only Reference

| Variant | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift25` | `+0.0469` | `+0.0000` | `+0.0156` |
| `por_actor_hidden_partial_shift50` | `-0.4219` | `-0.4688` | `-0.4531` |

## Interpretation

- This is a same-seed actor-hidden FiLM follow-up around the mild scale-only branch.
- A partial-shift variant only counts as interesting if it improves the lower band over mild scale-only without giving back the gap/shoulder tie.
