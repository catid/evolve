# Next-Round Baseline Sync

- archived frozen pack: `outputs/reports/frozen_benchmark_pack.json`
- live active benchmark pack: `outputs/reports/portfolio_candidate_pack.json`
- repaired current gate reference pack: `outputs/reports/round6_current_benchmark_pack.json`

## Archived Frozen Baseline

- retry-block KL learner-state `SARE` threshold: `0.3125`
- combined DoorKey KL learner-state `SARE` threshold: `0.7122`

## Active Round6 Benchmark

- candidate: `round6`
- retry-block KL learner-state `SARE` mean: `1.0000`
- frozen-comparable combined KL learner-state `SARE` mean: `1.0000`
- holdout SARE/token/single: `0.8320` / `0.8333` / `0.8333`
- healthy anti-regression SARE/token/single: `1.0000` / `0.9141` / `0.9167`
- route validation incumbent pass: `True`
- stability incumbent pass: `True`

## Current Frontier Priors

- default restart prior: `round7`
- replay-validated alternate: `round10`
- hold-only priors: `['round5']`
- retired priors: `['door3_post5', 'post_unlock_x5']`
- seed roles: `sentinel=SeedRole(lane='prospective_c', seed=193)`, `support=SeedRole(lane='prospective_f', seed=233)`, `weakness=SeedRole(lane='prospective_h', seed=269)`, `guardrail=SeedRole(lane='prospective_h', seed=277)`
