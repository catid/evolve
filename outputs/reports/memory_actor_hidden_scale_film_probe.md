# Memory Actor-Hidden Scale-Only FiLM Probe

- hypothesis: the hidden-state FiLM lift is real but the additive shift term is the unstable part; scale-only modulation may keep the lower/gap lift while improving the shoulder band
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Scale Only | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0` | `0.0000` | `0.0000` |
| `por_actor_hidden_film` | `0.0000` | `0.4375` | `0.4688` | `0.4531` | `0.0447` | `0.0333` | `0.0321` | `0.4978` | `0.7500` | `0.2889` | `0.1444` | `0` | `0.0287` | `0.0326` |
| `por_actor_hidden_scale_film` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.6280` | `1.6280` | `1.6280` | `0.3722` | `0.3913` | `0.3527` | `0.1764` | `1` | `0.0536` | `0.0000` |
| `por_actor_hidden_scale_film_mild` | `0.0000` | `0.4219` | `0.4688` | `0.4531` | `0.2755` | `0.2915` | `0.3231` | `0.2737` | `0.3333` | `0.3312` | `0.1159` | `1` | `0.0314` | `0.0000` |

## Deltas vs Base

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_film` | `+0.0000` | `+0.0625` | `+0.0156` | `-0.0156` |
| `por_actor_hidden_scale_film` | `+0.0000` | `-0.3750` | `-0.4531` | `-0.4688` |
| `por_actor_hidden_scale_film_mild` | `+0.0000` | `+0.0469` | `+0.0156` | `-0.0156` |

## Deltas vs Hidden-FiLM Reference

| Variant | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: |
| `por_actor_hidden_scale_film` | `-0.4375` | `-0.4688` | `-0.4531` |
| `por_actor_hidden_scale_film_mild` | `-0.0156` | `+0.0000` | `+0.0000` |

## Interpretation

- This is a same-seed actor-hidden FiLM follow-up, not a new family screen.
- The scale-only branch only counts as interesting if it preserves or improves the lower/gap band while recovering some of the shoulder loss from full hidden FiLM.
