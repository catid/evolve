# Memory Actor-Hidden Low-Margin FiLM Gate Probe

- hypothesis: the winning partial_shift25 branch may be over-applying the hidden-FiLM intervention; multiplying it by a base-policy low-margin gate may keep the greedy conversion while concentrating the intervention on ambiguous decisions
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Low-Margin Gate | Gate Mean | Margin Before | Shift Weight | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0.0000` | `0.0000` | `0.0000` | `0.00` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift25_low_margin35` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.0356` | `1.0356` | `1.0356` | `0.3831` | `0.4118` | `0.0003` | `0.0000` | `1.0355` | `0.25` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift25_low_margin50` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.3549` | `1.3549` | `1.3549` | `0.4247` | `0.4516` | `0.0000` | `0.0000` | `1.3548` | `0.25` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift25` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6550` | `2.6550` | `2.6550` | `0.4940` | `0.5067` | `0.0000` | `0.1240` | `0.0000` | `0.25` | `0.1230` | `0.0315` |

## Deltas vs Partial-Shift25 Reference

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift25_low_margin35` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift25_low_margin50` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |

## Interpretation

- This is a same-seed low-margin gating probe around the current `partial_shift25` architecture win.
- A low-margin-gated variant only counts as interesting if it preserves the new greedy lift while improving the sampled band or softening the shoulder tradeoff versus the ungated incumbent.
