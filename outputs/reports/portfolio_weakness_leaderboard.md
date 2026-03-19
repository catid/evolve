# Portfolio Weakness Seed Leaderboard

- source campaign: `lss_portfolio_campaign`
- git commit: `de91a953dee74128b91a939ac799f0415c7b6fab`
- git dirty: `True`
- bounded weakness seed: `prospective_h/269`
- active benchmark/control state: `round6=0.9844`, `token_dense=1.0000`, `single_expert=1.0000`, `classification=control_advantaged`

## Historical Leaderboard On The Weakness Seed

| Source | Candidate | Seed 269 | Seed 271 | Seed 277 | prospective_h Mean | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `migration` | `door3_post5` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | the broader campaign still failed to produce a migration-quality challenger |
| `stress_extended` | `round10` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | the broader campaign did not yield a meaningful post-control replacement |
| `migration` | `round5` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | the broader campaign still failed to produce a migration-quality challenger |
| `stress_extended` | `round5` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | the broader campaign did not yield a meaningful post-control replacement |
| `migration` | `round7` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | the broader campaign still failed to produce a migration-quality challenger |
| `stress_extended` | `round7` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | the broader campaign did not yield a meaningful post-control replacement |
| `migration` | `post_unlock_x5` | `1.0000` | `1.0000` | `0.8906` | `0.9635` | the broader campaign still failed to produce a migration-quality challenger |
| `migration` | `door2_post4` | `1.0000` | `1.0000` | `0.6094` | `0.8698` | the broader campaign still failed to produce a migration-quality challenger |
| `stress_extended` | `post_unlock_weighted` | `1.0000` | `1.0000` | `0.6094` | `0.8698` | the broader campaign did not yield a meaningful post-control replacement |
| `migration` | `carry2_post4` | `1.0000` | `1.0000` | `0.5000` | `0.8333` | the broader campaign still failed to produce a migration-quality challenger |
| `migration` | `disagree100` | `1.0000` | `1.0000` | `0.0000` | `0.6667` | the broader campaign still failed to produce a migration-quality challenger |
| `stress_extended` | `round12` | `0.9844` | `1.0000` | `1.0000` | `0.9948` | the broader campaign did not yield a meaningful post-control replacement |
| `stress_extended` | `round6` | `0.9844` | `1.0000` | `1.0000` | `0.9948` | the broader campaign did not yield a meaningful post-control replacement |
| `migration` | `post_unlock_x6` | `0.6250` | `1.0000` | `0.6094` | `0.7448` | the broader campaign still failed to produce a migration-quality challenger |
| `migration` | `disagree050` | `0.6250` | `1.0000` | `0.0000` | `0.5417` | the broader campaign still failed to produce a migration-quality challenger |
| `migration` | `carry3_post5` | `0.0000` | `1.0000` | `1.0000` | `0.6667` | the broader campaign still failed to produce a migration-quality challenger |
| `migration` | `round5_phase_balanced_dis050` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | the broader campaign still failed to produce a migration-quality challenger |
| `migration` | `round5_phase_balanced_dis100` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | the broader campaign still failed to produce a migration-quality challenger |

## Full Solvers Of The Weakness Seed

- candidates with `seed_269 = 1.0000`: `['carry2_post4', 'disagree100', 'door2_post4', 'door3_post5', 'post_unlock_weighted', 'post_unlock_x5', 'round10', 'round5', 'round7']`
- unique count: `9`

## Interpretation

- The bounded weakness is real but not unique to `round6`: several historical challengers already solve `prospective_h/269` cleanly.
- That makes `269` a useful target seed for future search, but not a sufficient promotion criterion by itself. The repo history already shows that local fixes here can still fail the broader post-control league.
- The right next-step use of this leaderboard is to bias future bounded search toward families that solve `269` without spending more budget on families that either stay at `0.9844` or collapse well below baseline.
