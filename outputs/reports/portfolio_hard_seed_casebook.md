# Portfolio Hard-Seed Casebook

- source campaign: `lss_portfolio_campaign`
- git commit: `ce1548efc7220bfce1d74defe4379e67d8df18ca`
- git dirty: `True`
- verified tie-bucket candidates: `['round10', 'round10_post_unlock_x4_dis025', 'round10_post_unlock_x5', 'round10_carry2_post4', 'round10_conf_post4', 'round10_door2_post4']`
- representative challenger for detailed curves: `round10`

## Summary

- `prospective_c/193` is a global hard-failure seed on the portfolio dev split, not a SARE-specific miss.
- All six fully verified tie-bucket challengers reproduce the exact same result on that seed: `SARE = 0.0`, matched `token_dense = 0.0`, matched `single_expert = 0.0`.
- Adjacent seeds `prospective_c/181` and `prospective_c/191` are solved by every verified challenger and every matched control, so the failure is highly localized rather than a broad `prospective_c` collapse.
- `prospective_f/233` remains a route/control differentiator: verified challengers and `single_expert` solve it at `1.0`, while matched `token_dense` stays at `0.625`.
- Future challenger screens should keep `prospective_c/193` as a hardness sentinel, but not treat it as evidence that one tied candidate is stronger than another unless some line actually breaks the all-zero control parity.

## Seed-Level Outcomes Across Verified Challengers

| Lane | Seed | Classification | Candidate | SARE | token_dense | single_expert |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| prospective_c | 181 | `shared_control_parity` | `round10` | `1.0000` | `1.0000` | `1.0000` |
| prospective_c | 181 | `shared_control_parity` | `round10_carry2_post4` | `1.0000` | `1.0000` | `1.0000` |
| prospective_c | 181 | `shared_control_parity` | `round10_conf_post4` | `1.0000` | `1.0000` | `1.0000` |
| prospective_c | 181 | `shared_control_parity` | `round10_door2_post4` | `1.0000` | `1.0000` | `1.0000` |
| prospective_c | 181 | `shared_control_parity` | `round10_post_unlock_x4_dis025` | `1.0000` | `1.0000` | `1.0000` |
| prospective_c | 181 | `shared_control_parity` | `round10_post_unlock_x5` | `1.0000` | `1.0000` | `1.0000` |
| prospective_c | 191 | `shared_control_parity` | `round10` | `1.0000` | `1.0000` | `1.0000` |
| prospective_c | 191 | `shared_control_parity` | `round10_carry2_post4` | `1.0000` | `1.0000` | `1.0000` |
| prospective_c | 191 | `shared_control_parity` | `round10_conf_post4` | `1.0000` | `1.0000` | `1.0000` |
| prospective_c | 191 | `shared_control_parity` | `round10_door2_post4` | `1.0000` | `1.0000` | `1.0000` |
| prospective_c | 191 | `shared_control_parity` | `round10_post_unlock_x4_dis025` | `1.0000` | `1.0000` | `1.0000` |
| prospective_c | 191 | `shared_control_parity` | `round10_post_unlock_x5` | `1.0000` | `1.0000` | `1.0000` |
| prospective_c | 193 | `global_hard_failure` | `round10` | `0.0000` | `0.0000` | `0.0000` |
| prospective_c | 193 | `global_hard_failure` | `round10_carry2_post4` | `0.0000` | `0.0000` | `0.0000` |
| prospective_c | 193 | `global_hard_failure` | `round10_conf_post4` | `0.0000` | `0.0000` | `0.0000` |
| prospective_c | 193 | `global_hard_failure` | `round10_door2_post4` | `0.0000` | `0.0000` | `0.0000` |
| prospective_c | 193 | `global_hard_failure` | `round10_post_unlock_x4_dis025` | `0.0000` | `0.0000` | `0.0000` |
| prospective_c | 193 | `global_hard_failure` | `round10_post_unlock_x5` | `0.0000` | `0.0000` | `0.0000` |
| prospective_f | 233 | `route_control_differentiator` | `round10` | `1.0000` | `0.6250` | `1.0000` |
| prospective_f | 233 | `route_control_differentiator` | `round10_carry2_post4` | `1.0000` | `0.6250` | `1.0000` |
| prospective_f | 233 | `route_control_differentiator` | `round10_conf_post4` | `1.0000` | `0.6250` | `1.0000` |
| prospective_f | 233 | `route_control_differentiator` | `round10_door2_post4` | `1.0000` | `0.6250` | `1.0000` |
| prospective_f | 233 | `route_control_differentiator` | `round10_post_unlock_x4_dis025` | `1.0000` | `0.6250` | `1.0000` |
| prospective_f | 233 | `route_control_differentiator` | `round10_post_unlock_x5` | `1.0000` | `0.6250` | `1.0000` |

## Round6 Mechanism Contrast

| Case | Round | Greedy | Post-Unlock Frac | Disagreement | Route Entropy | Path Entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `prospective_c:193` | `1` | `0.0000` | `0.0000` | `0.6071` | `1.3852` | `0.7761` |
| `prospective_c:193` | `2` | `0.0000` | `0.0000` | `0.4061` | `1.3827` | `0.4726` |
| `prospective_c:193` | `3` | `0.0000` | `0.0000` | `0.0002` | `1.3825` | `0.6180` |
| `prospective_c:193` | `4` | `0.0000` | `0.0000` | `0.0003` | `1.3825` | `0.6296` |
| `prospective_c:193` | `5` | `0.0000` | `0.0000` | `0.0002` | `1.3825` | `0.6743` |
| `prospective_c:193` | `6` | `0.0000` | `0.0000` | `0.0002` | `1.3825` | `0.6668` |
| `prospective_c:191` | `1` | `0.0000` | `0.0000` | `0.9947` | `1.3831` | `0.9544` |
| `prospective_c:191` | `2` | `0.0000` | `0.0000` | `0.9948` | `1.3824` | `1.3636` |
| `prospective_c:191` | `3` | `0.0000` | `0.0000` | `0.4905` | `1.3817` | `1.2891` |
| `prospective_c:191` | `4` | `0.0000` | `0.7507` | `0.3862` | `1.3780` | `1.3187` |
| `prospective_c:191` | `5` | `1.0000` | `0.9761` | `0.9683` | `1.3766` | `1.3317` |
| `prospective_c:191` | `6` | `1.0000` | `0.4288` | `0.0000` | `1.3764` | `1.3183` |
| `prospective_f:233` | `1` | `0.0000` | `0.0000` | `0.9995` | `1.3855` | `0.5385` |
| `prospective_f:233` | `2` | `0.0000` | `0.0000` | `0.9941` | `1.3783` | `0.8960` |
| `prospective_f:233` | `3` | `0.0000` | `0.0000` | `0.4924` | `1.3775` | `0.9706` |
| `prospective_f:233` | `4` | `0.0000` | `0.7666` | `0.4151` | `1.3752` | `0.9819` |
| `prospective_f:233` | `5` | `1.0000` | `0.9786` | `0.2237` | `1.3748` | `0.9874` |
| `prospective_f:233` | `6` | `1.0000` | `0.4515` | `0.0000` | `1.3768` | `1.0055` |

## Control Contrast On prospective_c/193

| Variant | Round | Greedy | Post-Unlock Frac | Disagreement | Route Entropy | Path Entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `token_dense` | `1` | `0.0000` | `0.3672` | `0.4486` | `0.0000` | `0.0000` |
| `token_dense` | `2` | `0.0000` | `0.0000` | `0.0338` | `0.0000` | `0.0000` |
| `token_dense` | `3` | `0.0000` | `0.0000` | `0.1881` | `0.0000` | `0.0000` |
| `token_dense` | `4` | `0.0000` | `0.0000` | `0.0158` | `0.0000` | `0.0000` |
| `token_dense` | `5` | `0.0000` | `0.0000` | `0.0469` | `0.0000` | `0.0000` |
| `token_dense` | `6` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `single_expert` | `1` | `0.0000` | `0.0000` | `1.0000` | `0.0000` | `0.0000` |
| `single_expert` | `2` | `0.0000` | `0.0000` | `0.3586` | `0.0000` | `0.0000` |
| `single_expert` | `3` | `0.0000` | `0.0000` | `0.0627` | `0.0000` | `0.0000` |
| `single_expert` | `4` | `0.0000` | `0.0000` | `0.0158` | `0.0000` | `0.0000` |
| `single_expert` | `5` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `single_expert` | `6` | `0.0000` | `0.0000` | `0.0469` | `0.0000` | `0.0000` |

## Interpretation

- `prospective_c/193` differs from the usual late-phase recovery pattern because `round6` never reaches post-unlock at all; its post-unlock fraction stays `0.0000` for every round and disagreement collapses to near-zero by round 3.
- The matched controls also fail there, which means the seed is not currently a useful discriminator between `round6` and its tied challengers.
- By contrast, solved seeds such as `prospective_c/191` and `prospective_f/233` show the expected recovery signature: high post-unlock occupancy appears in the successful rounds and greedy success turns on only after that phase is actually reached.
