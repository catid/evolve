# Portfolio Screening Spec

- source campaign: `lss_portfolio_campaign`
- git commit: `0cd012e8910aa460b142bfd58eabd19bca7850f2`
- git dirty: `True`
- restart default: `['round7']`
- reserve priors: `['round10', 'round5']`
- retired priors: `['door3_post5', 'post_unlock_x5']`

## Minimal Seed Roles

| Role | Lane | Seed | round6 | token_dense | single_expert | Classification | Use |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `sentinel` | `prospective_c` | 193 | `0.0000` | `0.0000` | `0.0000` | `global_hard_failure` | `track only; do not rank on this seed` |
| `ranking_support` | `prospective_f` | 233 | `1.0000` | `0.6250` | `1.0000` | `benchmark_support` | `prune if a candidate regresses below 1.0000` |
| `ranking_weakness` | `prospective_h` | 269 | `0.9844` | `1.0000` | `1.0000` | `control_advantaged` | `require improvement above round6 before broader dev spend` |
| `guardrail` | `prospective_h` | 277 | `1.0000` | `1.0000` | `1.0000` | `shared_control_parity` | `prune if a candidate falls below 1.0000` |

## Restart Frontier Under The Spec

| Candidate | Policy Bucket | support_233 | weakness_269 | guardrail_277 | Screen Rule |
| --- | --- | ---: | ---: | ---: | --- |
| `round7` | `restart_default` | `1.0000` | `1.0000` | `1.0000` | `advance_for_broader_dev` |
| `round10` | `reserve_same_signal_higher_cost` | `1.0000` | `1.0000` | `1.0000` | `advance_for_broader_dev` |
| `round5` | `reserve_below_incumbent` | `1.0000` | `1.0000` | `1.0000` | `advance_for_broader_dev` |
| `door3_post5` | `retire_structural_regression` | `0.4531` | `1.0000` | `1.0000` | `prune_support_regression` |
| `post_unlock_x5` | `retire_local_only_fix` | `n/a` | `1.0000` | `0.8906` | `prune_guardrail_regression` |

## Screening Policy

- global-hard sentinels already confirmed: `['prospective_c:193']`
- measured differentiator seeds: `['prospective_f:233']`
- Any bounded mini-sweep should include all four roles above before it gets a broader dev rerun.
- A candidate that regresses on `prospective_f/233` or `prospective_h/277` should be pruned immediately after rerun.
- A candidate that only ties `prospective_h/269` should not advance, because the measured frontier already shows that tie bucket does not survive broader post-control comparison.
- A candidate only earns broader dev/fairness budget if it keeps `f/233 = 1.0000`, keeps `h/277 = 1.0000`, and improves `h/269` above `round6 = 0.9844`.
