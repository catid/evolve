# Memory Actor-Hidden Post-Norm Probe

- hypothesis: the winning partial-shift branch may need the current shift/scale regime, but not the resulting hidden-state magnitude; a post-FiLM layer norm may preserve the greedy conversion while conditioning the actor hidden state better
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Post Norm | Gate Signal | Gate Mean | Post Hidden Norm | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift25_scale325_post_norm` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.9094` | `0.9094` | `0.9094` | `0.3871` | `0.6000` | `1` | `0.3325` | `0.1081` | `11.3186` | `0.0369` | `0.0072` |
| `por_actor_hidden_partial_shift25_post_norm` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.6953` | `1.6953` | `1.6953` | `0.3501` | `0.3846` | `1` | `0.2945` | `0.1031` | `11.3383` | `0.0255` | `0.0054` |
| `por_actor_hidden_partial_shift25` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6550` | `2.6550` | `2.6550` | `0.4940` | `0.5067` | `0` | `0.3544` | `0.1240` | `10.0578` | `0.1230` | `0.0315` |

## Deltas vs Partial-Shift25 Reference

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift25_post_norm` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift25_scale325_post_norm` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |

## Interpretation

- This is a same-seed post-FiLM normalization probe around the current `partial_shift25` architecture win.
- A post-norm variant only counts as interesting if it preserves the new greedy lift while improving the sampled band or reducing the conditioning pathologies around the winner.
