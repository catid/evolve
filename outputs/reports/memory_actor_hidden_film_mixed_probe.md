# Memory Actor-Hidden FiLM Mixed-Gate Probe

- hypothesis: the hidden-state FiLM win is real, but raw duration gate pushes too hard into the shoulder band; mixing duration and stability should preserve the lower-band lift while reducing shoulder damage
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Duration Mix | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.00` | `0.0000` | `0.0000` |
| `por_actor_hidden_film` | `0.0000` | `0.4375` | `0.4688` | `0.4531` | `0.0447` | `0.0333` | `0.0321` | `0.4978` | `0.7500` | `0.2889` | `0.1444` | `1.00` | `0.0287` | `0.0326` |
| `por_actor_hidden_film_mixed50` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `2.2710` | `2.2710` | `2.2710` | `0.3619` | `0.3846` | `0.1836` | `0.0918` | `0.50` | `0.0341` | `0.0635` |
| `por_actor_hidden_film_mixed75` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.5565` | `1.5565` | `1.5565` | `0.4227` | `0.5000` | `0.2450` | `0.1225` | `0.75` | `0.0306` | `0.0432` |

## Deltas vs Base

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_film` | `+0.0000` | `+0.0625` | `+0.0156` | `-0.0156` |
| `por_actor_hidden_film_mixed50` | `+0.0000` | `-0.3750` | `-0.4531` | `-0.4688` |
| `por_actor_hidden_film_mixed75` | `+0.0000` | `-0.3750` | `-0.4531` | `-0.4688` |

## Deltas vs Hidden-FiLM Reference

| Variant | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: |
| `por_actor_hidden_film_mixed50` | `-0.4375` | `-0.4688` | `-0.4531` |
| `por_actor_hidden_film_mixed75` | `-0.4375` | `-0.4688` | `-0.4531` |

## Interpretation

- This is a same-seed architecture follow-up, not a new family screen.
- The mixed gate only counts as a real improvement if it retains the lower/gap lift from hidden FiLM while improving or at least preserving the shoulder band.
