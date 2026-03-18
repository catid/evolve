# Long Campaign Baseline Sync

- frozen manifest: `configs/claims/doorkey_frozen_claim.yaml`
- combined baseline csv: `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.csv`
- weak-block baseline csv: `outputs/reports/lss_final_block_single_expert_control_report.csv`

## Weak Block

| Seed | KL learner-state token_dense | KL learner-state single_expert | KL learner-state SARE |
| --- | ---: | ---: | ---: |
| 47 | 1.0 | 0.453125 | 0.0 |
| 53 | 1.0 | 0.515625 | 0.515625 |
| 59 | 1.0 | 0.421875 | 0.421875 |

## Frozen Thresholds

- retry-block KL learner-state `SARE` mean: `0.3125`
- retry-block KL learner-state `single_expert` mean: `0.4635`
- combined KL learner-state `SARE` mean: `0.7122`

## Interpretation

- The frozen retry block and combined DoorKey picture still match the current sealed claim envelope closely enough to start the staged campaign from the accepted baseline.
