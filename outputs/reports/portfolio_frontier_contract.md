# Portfolio Frontier Contract

- source campaign: `lss_portfolio_campaign`
- git commit: `a9d089c1aaaf11bdbe569a92684ca96676c4aed3`
- git dirty: `True`
- consistency overall: `pass`

## Benchmark State

- active benchmark: `round6`
- active pack: `outputs/reports/portfolio_candidate_pack.json`
- archived frozen pack: `outputs/reports/frozen_benchmark_pack.json`

## Frontier Roles

- default restart prior: `round7`
- replay-validated alternate: `round10`
- hold-only priors: `['round5']`
- retired priors: `['door3_post5', 'post_unlock_x5']`

## Seed Contract

- seed roles: `{'guardrail': {'lane': 'prospective_h', 'seed': 277}, 'ranking_support': {'lane': 'prospective_f', 'seed': 233}, 'ranking_weakness': {'lane': 'prospective_h', 'seed': 269}, 'sentinel': {'lane': 'prospective_c', 'seed': 193}}`
- global-hard labels: `['prospective_c:193']`
- differentiator labels: `['prospective_f:233']`
- thresholds: `{'support_min_success': 1.0, 'weakness_min_success_exclusive': 0.984375001, 'guardrail_min_success': 1.0}`
- support status: `measured_support_regression`

## Interpretation

- This contract is the single authoritative snapshot of the current measured portfolio frontier.
- It freezes the active benchmark, restart prior, measured alternate, retired set, and seed-level screening contract in one place.
- Future bounded search or doc updates should be treated as drifting until this contract is intentionally refreshed.
