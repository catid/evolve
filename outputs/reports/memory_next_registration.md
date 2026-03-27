# Memory Next Registration

- target substantive runs: `80`
- 50/50 program split:
  - exploit track: `40` substantive candidates
  - exploration track: `40` substantive candidates

## Memory Groups

- `dev_groups`:
  - `dev_a`: reset seeds `5000`–`5031`
  - `dev_b`: reset seeds `6000`–`6031`
  - `dev_c`: reset seeds `7000`–`7031`
- `holdout_groups`:
  - `holdout_a`: reset seeds `8000`–`8031`
  - `holdout_b`: reset seeds `9000`–`9031`
  - `holdout_c`: reset seeds `10000`–`10031`
- `healthy_groups`:
  - `healthy_upper_a`: reset seeds `11000`–`11031`
  - `healthy_upper_b`: reset seeds `12000`–`12031`
- `stability_groups`:
  - `stability_a`: reset seeds `13000`–`13031`
  - `stability_b`: reset seeds `14000`–`14031`

## Temperature Bands

- lower boundary: `[0.04, 0.045, 0.05]`
- strongest-gap band: `[0.05, 0.0525, 0.055]`
- shoulder fade-out: `[0.06, 0.065, 0.07]`
- upper plateau reference: `[0.075, 0.08, 0.09]`

## Families

- exploit:
  - `actor_hidden_continuation`: `Actor-Hidden FiLM Continuation` with `10` variants
  - `sampled_to_greedy_decode`: `Sampled-to-Greedy Decode` with `10` variants
  - `checkpoint_selection`: `Checkpoint / Selection` with `10` variants
  - `small_branch_consensus`: `Small Branch / Consensus` with `10` variants
- explore:
  - `sampled_success`: `Sampled-Success Distillation`
  - `selective_compute`: `Selective Compute-Structure Extraction`
  - `temperature_aware_teacher`: `Temperature-Aware Teacher`
  - `hybrid`: `Hybrid Practicalization`
  - `architecture_pilot`: `Architecture-Adjacent Memory Pilot`

## Decision Rules

- Stage B and Stage C both use calibration plus fresh-root rerun before a family is judged alive or dead.
- Stage D advances only candidates that rerun directionally, hold on Memory holdout, and avoid obvious healthy/stability regressions.
- Nothing changes in the accepted benchmark state unless a Memory candidate survives fairness, holdout, stability, and the final candidate-pack / gate path.
