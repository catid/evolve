# Memory Actor-Hidden FiLM Probe

- hypothesis: option-conditioned modulation works better earlier in the actor hidden representation than as direct logit or top2 output surgery, especially when driven by the raw duration gate
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_film` | `0.0000` | `0.4375` | `0.4688` | `0.4531` | `0.0447` | `0.0333` | `0.0321` | `0.4978` | `0.7500` | `0.2889` | `0.1444` | `0.0287` | `0.0326` |
| `por_actor_hidden_film_small` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.8312` | `0.8312` | `0.8312` | `0.2540` | `0.4286` | `0.2893` | `0.0723` | `0.0204` | `0.0313` |
| `gru_base` | `0.0000` | `0.2500` | `0.3594` | `0.4531` | `0.2449` | `0.2341` | `0.2284` | `0.2570` | `0.3333` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |

## POR Deltas vs Base

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_film` | `+0.0000` | `+0.0625` | `+0.0156` | `-0.0156` |
| `por_actor_hidden_film_small` | `+0.0000` | `-0.3750` | `-0.4531` | `-0.4688` |

## Interpretation

- This is a fresh architecture probe, not a checkpoint reuse sweep.
- The branch only counts as interesting if actor-hidden option conditioning preserves or improves the live sampled band relative to the matched POR base rerun.
