# Portfolio Weakness Shortlist

- source campaign: `lss_portfolio_campaign`
- git commit: `21eb925e1b7b348f79501ae35e8ef4f0ee78c503`
- git dirty: `True`
- bounded weakness seed: `prospective_h/269`

## Aggregated Historical Priors

| Candidate | Best Seed 269 | Best prospective_h Mean | Min Seed 277 | Sources | Bucket |
| --- | ---: | ---: | ---: | --- | --- |
| `round5` | `1.0000` | `1.0000` | `1.0000` | `['migration', 'stress_extended']` | `conservative_round_prior` |
| `round7` | `1.0000` | `1.0000` | `1.0000` | `['migration', 'stress_extended']` | `conservative_round_prior` |
| `door3_post5` | `1.0000` | `1.0000` | `1.0000` | `['migration']` | `structural_full_lane_prior` |
| `round10` | `1.0000` | `1.0000` | `1.0000` | `['stress_extended']` | `conservative_round_prior` |
| `post_unlock_x5` | `1.0000` | `0.9635` | `0.8906` | `['migration']` | `partial_revisit` |
| `door2_post4` | `1.0000` | `0.8698` | `0.6094` | `['migration']` | `local_only_fix` |
| `post_unlock_weighted` | `1.0000` | `0.8698` | `0.6094` | `['stress_extended']` | `local_only_fix` |
| `carry2_post4` | `1.0000` | `0.8333` | `0.5000` | `['migration']` | `local_only_fix` |
| `disagree100` | `1.0000` | `0.6667` | `0.0000` | `['migration']` | `local_only_fix` |
| `round12` | `0.9844` | `0.9948` | `1.0000` | `['stress_extended']` | `incumbent_parity` |
| `round6` | `0.9844` | `0.9948` | `1.0000` | `['stress_extended']` | `incumbent_parity` |
| `post_unlock_x6` | `0.6250` | `0.7448` | `0.6094` | `['migration']` | `dead_end` |
| `disagree050` | `0.6250` | `0.5417` | `0.0000` | `['migration']` | `dead_end` |
| `carry3_post5` | `0.0000` | `0.6667` | `1.0000` | `['migration']` | `dead_end` |
| `round5_phase_balanced_dis050` | `0.0000` | `0.0000` | `0.0000` | `['migration']` | `dead_end` |
| `round5_phase_balanced_dis100` | `0.0000` | `0.0000` | `0.0000` | `['migration']` | `dead_end` |

## Search Prior

- conservative round-count priors: `['round5', 'round7', 'round10']`
- structural full-lane priors: `['door3_post5']`
- partial revisits worth considering only after the full-lane priors: `['post_unlock_x5']`
- local-only fixes to deprioritize: `['door2_post4', 'post_unlock_weighted', 'carry2_post4', 'disagree100']`
- clear dead ends on the weakness surface: `['post_unlock_x6', 'disagree050', 'carry3_post5', 'round5_phase_balanced_dis050', 'round5_phase_balanced_dis100']`

## Interpretation

- The cleanest bounded-search priors are the simple round-count families that solve `prospective_h/269` and keep the whole `prospective_h` lane at `1.0000` without additional mechanism complexity.
- Structural one-offs such as `door3_post5` also solve the lane cleanly, but they are less conservative starting points because they add a different mechanism family and still failed broader league selection.
- Local-only fixes that solve `269` while degrading `277` should not be treated as first-line priors; they are useful only if the conservative full-lane priors are exhausted.
- Future challenger work can now start from a narrow evidence-backed shortlist instead of replaying the entire historical tie bucket.
