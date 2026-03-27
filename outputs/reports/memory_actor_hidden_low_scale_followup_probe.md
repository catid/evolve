# Memory Actor-Hidden Low-Scale Follow-Up Probe

- hypothesis: the new `partial_shift225` winner may want slightly less hidden-FiLM scale than `0.35`; a softer scale could preserve greedy conversion while improving lower or gap behavior
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Gate Mean | Shift Weight | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift225` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6272` | `2.6143` | `2.5712` | `0.4279` | `0.4643` | `0.3340` | `0.1169` | `0.2250` | `0.0987` | `0.0224` |
| `por_actor_hidden_partial_shift225_scale325` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.0609` | `1.0609` | `1.0609` | `0.2219` | `0.5000` | `0.4561` | `0.1482` | `0.2250` | `0.0592` | `0.0117` |
| `por_actor_hidden_partial_shift225_scale30` | `0.0000` | `0.5469` | `0.5156` | `0.5156` | `0.1892` | `0.3105` | `0.3762` | `0.5017` | `0.5238` | `0.3313` | `0.0994` | `0.2250` | `0.0835` | `0.0148` |

## Interpretation

- incumbent `partial_shift225` remains the reference greedy-conversion point at `0.4688` greedy with sampled `0.4688 / 0.4688 / 0.4688`
- `scale325` changes the same surface to `0.0000 / 0.0000 / 0.0000 / 0.0000` on greedy/lower/gap/shoulder
- `scale30` changes the same surface to `0.0000 / 0.5469 / 0.5156 / 0.5156` on greedy/lower/gap/shoulder

- conclusion: at least one lower-scale variant survives or improves on the incumbent and needs a longer confirmation pass
