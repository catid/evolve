# Memory Next Band Definition

- current best local greedy-conversion point: `partial_shift22`

## Temperature Bands

- lower boundary band: `[0.04, 0.045, 0.05]`
- strongest-gap band: `[0.05, 0.0525, 0.055]`
- shoulder / fade-out band: `[0.06, 0.065, 0.07]`
- upper plateau reference: `[0.075, 0.08, 0.09]`

## Evaluation Splits

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

## Interpretation

- The Memory branch is now registered as a real benchmark family with explicit lower, gap, and shoulder bands rather than one attractive curve.
- `partial_shift22` is the actor-hidden local greedy-conversion incumbent; `POR switchy seed7` and `gru_long_11` remain the non-practicalized baselines for fair comparison.
