# Memory Actor-Hidden Blend Probe

- hypothesis: the winning partial_shift25 branch may need the transformed hidden state only on a subset of states; a learned blend gate could preserve greedy conversion while letting the policy fall back toward the base hidden state where the transformation hurts
- task: `MiniGrid-MemoryS9-v0`
- fresh matched runs: `4`
- same-seed architecture isolate: `seed=7` for all runs
- evaluation episodes per mode: `64`

## Aggregate

| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Blend Gate | Blend Scale | Blend Mean | Gate Signal | Gate Mean | Scale Norm | Shift Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `por_base` | `0.0000` | `0.3750` | `0.4531` | `0.4688` | `0.2994` | `0.4250` | `0.7277` | `0.5172` | `0.5303` | `0` | `0.00` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `por_actor_hidden_partial_shift25_blend100` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.7835` | `0.7835` | `0.7835` | `0.1926` | `0.3333` | `1` | `1.00` | `0.8818` | `0.3077` | `0.1077` | `0.0430` | `0.0095` |
| `por_actor_hidden_partial_shift25_blend150` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.0690` | `1.0690` | `1.0690` | `0.3207` | `0.3333` | `1` | `1.50` | `1.0000` | `0.3145` | `0.1101` | `0.0333` | `0.0068` |
| `por_actor_hidden_partial_shift25` | `0.4688` | `0.4688` | `0.4688` | `0.4688` | `2.6550` | `2.6550` | `2.6550` | `0.4940` | `0.5067` | `0` | `1.00` | `0.0000` | `0.3544` | `0.1240` | `0.1230` | `0.0315` |

## Deltas vs Partial-Shift25 Reference

| Variant | Delta Greedy | Delta Lower | Delta Gap | Delta Shoulder |
| --- | ---: | ---: | ---: | ---: |
| `por_actor_hidden_partial_shift25_blend100` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |
| `por_actor_hidden_partial_shift25_blend150` | `-0.4688` | `-0.4688` | `-0.4688` | `-0.4688` |

## Interpretation

- This is a same-seed learned blend-gate probe around the current `partial_shift25` architecture win.
- A blend-gated variant only counts as interesting if it preserves the new greedy lift while improving the sampled band or softening the shoulder tradeoff versus the ungated incumbent.
