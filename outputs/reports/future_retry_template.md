# Future Retry Template

Use this template before any future DoorKey retry is run. Do not skip any field.

## Mechanism Hypothesis

- hypothesis:
- evidence source report(s):
- why this is one mechanism rather than a broad search:

## Exact Intervention

- intervention family (must stay inside current KL learner-state family):
- exact config path(s):
- exact CLI:

## Fixed Seeds And Controls

- retry-block seeds: `[47, 53, 59]`
- required controls on the same block:
  - recovered token_dense
  - KL learner-state token_dense
  - KL learner-state single_expert
  - baseline PPO SARE

## Success Bar

- candidate KL learner-state SARE retry-block mean must exceed `0.3125`
- candidate KL learner-state SARE must at least match matched KL learner-state single_expert on the same block
- candidate combined DoorKey KL learner-state SARE mean must stay at or above `0.7122`

## Failure Bar

- any missing fairness control
- any non-external-64 evaluation path
- any disallowed claim widening
- any regression below the frozen combined DoorKey picture

## Required Outputs

- retry report markdown + csv
- updated combined DoorKey report markdown + csv
- claim gate report markdown + json

## Claim-Gate Invocation

Run the candidate through the automated gate using the manifest `configs/claims/doorkey_frozen_claim.yaml` before any public claim update.

```bash
./scripts/run_claim_gate.sh \
  configs/claims/doorkey_frozen_claim.yaml \
  <candidate-json> \
  outputs/reports/claim_gate_retry_check.md \
  outputs/reports/claim_gate_retry_check.json
```
