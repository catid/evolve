# Memory Next Source Quality

- diagnosis: `both, with decode/selection primary`
- strongest sampled-to-greedy gap: `0.4688`
- POR lower-band edge over strongest recurrent control: `0.3750`
- actor-hidden local greedy point (`partial_shift22`) : greedy `0.5208`, gap-band `0.5208`

## Audit Table

| Checkpoint | Mode | Success | Return | Max Prob | Margin | Option Duration | Option Switch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gru_long_11 | gap_t055 | 0.4271 | 0.2482 | 0.3573 | 0.2338 | 0.0000 | 0.0000 |
| gru_long_11 | greedy | 0.0000 | 0.0000 | 0.3593 | 0.2464 | 0.0000 | 0.0000 |
| gru_long_11 | lower_t005 | 0.3646 | 0.1983 | 0.3574 | 0.2345 | 0.0000 | 0.0000 |
| gru_long_11 | shoulder_t008 | 0.5208 | 0.4349 | 0.3545 | 0.2232 | 0.0000 | 0.0000 |
| partial_shift22 | gap_t055 | 0.5208 | 0.5132 | 0.6490 | 1.3160 | 2.4995 | 0.1806 |
| partial_shift22 | greedy | 0.5208 | 0.5141 | 0.6654 | 1.4051 | 2.4234 | 0.1967 |
| partial_shift22 | lower_t005 | 0.5208 | 0.5132 | 0.6490 | 1.3160 | 2.4995 | 0.1806 |
| partial_shift22 | shoulder_t008 | 0.5208 | 0.5132 | 0.6485 | 1.3133 | 2.4994 | 0.1802 |
| por_nearby_base_11 | gap_t055 | 0.0000 | 0.0000 | 0.5831 | 1.6439 | 1.3804 | 0.0025 |
| por_nearby_base_11 | greedy | 0.0000 | 0.0000 | 0.5831 | 1.6439 | 1.3804 | 0.0025 |
| por_nearby_base_11 | lower_t005 | 0.0000 | 0.0000 | 0.5831 | 1.6439 | 1.3804 | 0.0025 |
| por_nearby_base_11 | shoulder_t008 | 0.0000 | 0.0000 | 0.5831 | 1.6439 | 1.3804 | 0.0025 |
| por_nearby_switchy_7 | gap_t055 | 0.4688 | 0.3209 | 0.4073 | 0.4295 | 2.8598 | 0.0332 |
| por_nearby_switchy_7 | greedy | 0.0000 | 0.0000 | 0.3774 | 0.2997 | 2.9211 | 0.0049 |
| por_nearby_switchy_7 | lower_t005 | 0.3750 | 0.2413 | 0.3936 | 0.3700 | 2.8925 | 0.0182 |
| por_nearby_switchy_7 | shoulder_t008 | 0.5208 | 0.4683 | 0.4706 | 0.7051 | 2.7066 | 0.1023 |

## Probe Table

| Checkpoint | Target | Samples | Classes | Test Acc | Majority | Lift | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gru_long_11 | consensus_action | 22259 | 7 | 0.8650 | 0.8612 | 0.0038 | completed |
| gru_long_11 | success_bucket | 22259 | 2 | 0.8021 | 0.8077 | -0.0056 | completed |
| partial_shift22 | consensus_action | 576 | 3 | 0.9914 | 0.7882 | 0.2032 | completed |
| partial_shift22 | success_bucket | 576 | 2 | 0.5431 | 0.5590 | -0.0159 | completed |
| por_nearby_base_11 | consensus_action | 38880 | 4 | 0.9982 | 0.9986 | -0.0004 | completed |
| por_nearby_base_11 | success_bucket | 38880 | 1 | 0.0000 | 0.0000 | 0.0000 | single_class |
| por_nearby_base_11 | switch_bucket | 38880 | 2 | 1.0000 | 0.9975 | 0.0025 | completed |
| por_nearby_base_11 | duration_bucket | 38880 | 1 | 0.0000 | 0.0000 | 0.0000 | single_class |
| por_nearby_switchy_7 | consensus_action | 20017 | 6 | 0.8539 | 0.8588 | -0.0049 | completed |
| por_nearby_switchy_7 | success_bucket | 20017 | 2 | 0.7952 | 0.7950 | 0.0002 | completed |
| por_nearby_switchy_7 | switch_bucket | 20017 | 2 | 1.0000 | 0.9904 | 0.0096 | completed |
| por_nearby_switchy_7 | duration_bucket | 20017 | 1 | 0.0000 | 0.0000 | 0.0000 | single_class |

## Interpretation

- The primary diagnosis is `both, with decode/selection primary`.
- Low-temperature sampled success is real on the best POR and recurrent checkpoints, so the Memory branch is not a missing-signal story.
- The probe fits show that persistence-related state is already exposed enough to decode shallow targets from frozen activations, but exact greedy still fails on the non-practicalized lines.
- `partial_shift22` demonstrates that the branch is not an absolute state ceiling, but the remaining gap still looks dominated by decode/selection and objective-shaping limits.
