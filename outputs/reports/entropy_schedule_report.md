# Entropy Schedule Report

- episodes per mode: `32`
- run count: `14`

## Best Schedules By Variant

| Variant | Schedule | Greedy Success | Best Sampled Success | Greedy Max Prob | Greedy Margin | Train Return |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sare | constant:0.001 | 0.0000 | 0.1875 | 0.8374 | 2.2396 | 0.0000 |
| single_expert | constant:0.01 | 0.0000 | 0.4375 | 0.3054 | 0.2648 | 0.2805 |

## Interpretation

- `sare` does not recover a nonzero greedy policy under any tested schedule. The best schedule `constant:0.001` still has greedy success `0.0000` while best sampled success is `0.1875`.
- `single_expert` does not recover a nonzero greedy policy under any tested schedule. The best schedule `constant:0.01` still has greedy success `0.0000` while best sampled success is `0.4375`.

## Schedule Table

| Variant | Schedule | Mode | Eval Success | Eval Return | Eval Max Prob | Eval Margin | Train Return | Entropy Coef | Throughput | Run |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sare | constant:0.0003 | greedy | 0.0000 | 0.0000 | 0.3137 | 0.4170 | 0.0000 | 0.0003 | 833.8252 | `minigrid_doorkey_sare_ent3e4` |
| sare | constant:0.0003 | sampled_t0.5 | 0.0000 | 0.0000 | 0.5082 | 0.4150 | 0.0000 | 0.0003 | 833.8252 | `minigrid_doorkey_sare_ent3e4` |
| sare | constant:0.0003 | sampled_t0.7 | 0.0000 | 0.0000 | 0.3983 | 0.4170 | 0.0000 | 0.0003 | 833.8252 | `minigrid_doorkey_sare_ent3e4` |
| sare | constant:0.0003 | sampled_t1.0 | 0.0312 | 0.0139 | 0.3138 | 0.4167 | 0.0000 | 0.0003 | 833.8252 | `minigrid_doorkey_sare_ent3e4` |
| sare | constant:0.001 | greedy | 0.0000 | 0.0000 | 0.8374 | 2.2396 | 0.0000 | 0.0010 | 845.2321 | `minigrid_doorkey_sare_ent1e3` |
| sare | constant:0.001 | sampled_t0.5 | 0.0000 | 0.0000 | 0.9847 | 2.2382 | 0.0000 | 0.0010 | 845.2321 | `minigrid_doorkey_sare_ent1e3` |
| sare | constant:0.001 | sampled_t0.7 | 0.0312 | 0.0056 | 0.9410 | 2.2426 | 0.0000 | 0.0010 | 845.2321 | `minigrid_doorkey_sare_ent1e3` |
| sare | constant:0.001 | sampled_t1.0 | 0.1875 | 0.0667 | 0.8379 | 2.2399 | 0.0000 | 0.0010 | 845.2321 | `minigrid_doorkey_sare_ent1e3` |
| sare | constant:0.003 | greedy | 0.0000 | 0.0000 | 0.9056 | 3.1139 | 0.0000 | 0.0030 | 828.4812 | `minigrid_doorkey_sare_ent3e3` |
| sare | constant:0.003 | sampled_t0.5 | 0.0000 | 0.0000 | 0.9967 | 3.1130 | 0.0000 | 0.0030 | 828.4812 | `minigrid_doorkey_sare_ent3e3` |
| sare | constant:0.003 | sampled_t0.7 | 0.0000 | 0.0000 | 0.9775 | 3.1135 | 0.0000 | 0.0030 | 828.4812 | `minigrid_doorkey_sare_ent3e3` |
| sare | constant:0.003 | sampled_t1.0 | 0.0000 | 0.0000 | 0.9054 | 3.1127 | 0.0000 | 0.0030 | 828.4812 | `minigrid_doorkey_sare_ent3e3` |
| sare | constant:0.01 | greedy | 0.0000 | 0.0000 | 0.3844 | 0.3060 | 0.0000 | 0.0100 | 839.6813 | `minigrid_doorkey_sare` |
| sare | constant:0.01 | sampled_t0.5 | 0.0000 | 0.0000 | 0.5876 | 0.3058 | 0.0000 | 0.0100 | 839.6813 | `minigrid_doorkey_sare` |
| sare | constant:0.01 | sampled_t0.7 | 0.0000 | 0.0000 | 0.4849 | 0.3057 | 0.0000 | 0.0100 | 839.6813 | `minigrid_doorkey_sare` |
| sare | constant:0.01 | sampled_t1.0 | 0.0000 | 0.0000 | 0.3844 | 0.3062 | 0.0000 | 0.0100 | 839.6813 | `minigrid_doorkey_sare` |
| sare | late_linear:0.01->0.001@0.75 | greedy | 0.0000 | 0.0000 | 0.5056 | 0.7131 | 0.0000 | 0.0010 | 845.1335 | `minigrid_doorkey_sare_latedrop_1e2_to_1e3` |
| sare | late_linear:0.01->0.001@0.75 | sampled_t0.5 | 0.0000 | 0.0000 | 0.7721 | 0.7130 | 0.0000 | 0.0010 | 845.1335 | `minigrid_doorkey_sare_latedrop_1e2_to_1e3` |
| sare | late_linear:0.01->0.001@0.75 | sampled_t0.7 | 0.0000 | 0.0000 | 0.6453 | 0.7129 | 0.0000 | 0.0010 | 845.1335 | `minigrid_doorkey_sare_latedrop_1e2_to_1e3` |
| sare | late_linear:0.01->0.001@0.75 | sampled_t1.0 | 0.0000 | 0.0000 | 0.5055 | 0.7128 | 0.0000 | 0.0010 | 845.1335 | `minigrid_doorkey_sare_latedrop_1e2_to_1e3` |
| sare | linear:0.003->0.0003 | greedy | 0.0000 | 0.0000 | 0.5264 | 0.2589 | 0.0000 | 0.0003 | 850.5307 | `minigrid_doorkey_sare_linear_3e3_to_3e4` |
| sare | linear:0.003->0.0003 | sampled_t0.5 | 0.0000 | 0.0000 | 0.6219 | 0.2531 | 0.0000 | 0.0003 | 850.5307 | `minigrid_doorkey_sare_linear_3e3_to_3e4` |
| sare | linear:0.003->0.0003 | sampled_t0.7 | 0.0000 | 0.0000 | 0.5797 | 0.2554 | 0.0000 | 0.0003 | 850.5307 | `minigrid_doorkey_sare_linear_3e3_to_3e4` |
| sare | linear:0.003->0.0003 | sampled_t1.0 | 0.0000 | 0.0000 | 0.5227 | 0.2448 | 0.0000 | 0.0003 | 850.5307 | `minigrid_doorkey_sare_linear_3e3_to_3e4` |
| sare | linear:0.01->0.001 | greedy | 0.0000 | 0.0000 | 0.3667 | 0.3814 | 0.0000 | 0.0010 | 837.0005 | `minigrid_doorkey_sare_linear_1e2_to_1e3` |
| sare | linear:0.01->0.001 | sampled_t0.5 | 0.0000 | 0.0000 | 0.5548 | 0.3807 | 0.0000 | 0.0010 | 837.0005 | `minigrid_doorkey_sare_linear_1e2_to_1e3` |
| sare | linear:0.01->0.001 | sampled_t0.7 | 0.0000 | 0.0000 | 0.4558 | 0.3809 | 0.0000 | 0.0010 | 837.0005 | `minigrid_doorkey_sare_linear_1e2_to_1e3` |
| sare | linear:0.01->0.001 | sampled_t1.0 | 0.0000 | 0.0000 | 0.3668 | 0.3808 | 0.0000 | 0.0010 | 837.0005 | `minigrid_doorkey_sare_linear_1e2_to_1e3` |
| single_expert | constant:0.0003 | greedy | 0.0000 | 0.0000 | 0.7617 | 1.4793 | 0.0000 | 0.0003 | 1402.3267 | `minigrid_doorkey_single_expert_ent3e4` |
| single_expert | constant:0.0003 | sampled_t0.5 | 0.0000 | 0.0000 | 0.9489 | 1.4809 | 0.0000 | 0.0003 | 1402.3267 | `minigrid_doorkey_single_expert_ent3e4` |
| single_expert | constant:0.0003 | sampled_t0.7 | 0.0000 | 0.0000 | 0.8793 | 1.4821 | 0.0000 | 0.0003 | 1402.3267 | `minigrid_doorkey_single_expert_ent3e4` |
| single_expert | constant:0.0003 | sampled_t1.0 | 0.0000 | 0.0000 | 0.7618 | 1.4811 | 0.0000 | 0.0003 | 1402.3267 | `minigrid_doorkey_single_expert_ent3e4` |
| single_expert | constant:0.001 | greedy | 0.0000 | 0.0000 | 0.4430 | 0.2716 | 0.0000 | 0.0010 | 1373.6469 | `minigrid_doorkey_single_expert_ent1e3` |
| single_expert | constant:0.001 | sampled_t0.5 | 0.0000 | 0.0000 | 0.6004 | 0.2699 | 0.0000 | 0.0010 | 1373.6469 | `minigrid_doorkey_single_expert_ent1e3` |
| single_expert | constant:0.001 | sampled_t0.7 | 0.0312 | 0.0038 | 0.5254 | 0.2686 | 0.0000 | 0.0010 | 1373.6469 | `minigrid_doorkey_single_expert_ent1e3` |
| single_expert | constant:0.001 | sampled_t1.0 | 0.0000 | 0.0000 | 0.4429 | 0.2693 | 0.0000 | 0.0010 | 1373.6469 | `minigrid_doorkey_single_expert_ent1e3` |
| single_expert | constant:0.003 | greedy | 0.0000 | 0.0000 | 0.4617 | 0.1608 | 0.0000 | 0.0030 | 1415.3295 | `minigrid_doorkey_single_expert_ent3e3` |
| single_expert | constant:0.003 | sampled_t0.5 | 0.0000 | 0.0000 | 0.5672 | 0.1586 | 0.0000 | 0.0030 | 1415.3295 | `minigrid_doorkey_single_expert_ent3e3` |
| single_expert | constant:0.003 | sampled_t0.7 | 0.0000 | 0.0000 | 0.5220 | 0.1580 | 0.0000 | 0.0030 | 1415.3295 | `minigrid_doorkey_single_expert_ent3e3` |
| single_expert | constant:0.003 | sampled_t1.0 | 0.0000 | 0.0000 | 0.4616 | 0.1600 | 0.0000 | 0.0030 | 1415.3295 | `minigrid_doorkey_single_expert_ent3e3` |
| single_expert | constant:0.01 | greedy | 0.0000 | 0.0000 | 0.3054 | 0.2648 | 0.2805 | 0.0100 | 1354.9022 | `minigrid_doorkey_single_expert` |
| single_expert | constant:0.01 | sampled_t0.5 | 0.4375 | 0.1786 | 0.4393 | 0.2670 | 0.2805 | 0.0100 | 1354.9022 | `minigrid_doorkey_single_expert` |
| single_expert | constant:0.01 | sampled_t0.7 | 0.4062 | 0.2099 | 0.3694 | 0.2685 | 0.2805 | 0.0100 | 1354.9022 | `minigrid_doorkey_single_expert` |
| single_expert | constant:0.01 | sampled_t1.0 | 0.4062 | 0.2301 | 0.3064 | 0.2680 | 0.2805 | 0.0100 | 1354.9022 | `minigrid_doorkey_single_expert` |
| single_expert | late_linear:0.01->0.001@0.75 | greedy | 0.0000 | 0.0000 | 0.3270 | 0.2947 | 0.1100 | 0.0010 | 1438.9466 | `minigrid_doorkey_single_expert_latedrop_1e2_to_1e3` |
| single_expert | late_linear:0.01->0.001@0.75 | sampled_t0.5 | 0.4062 | 0.1886 | 0.5451 | 0.3999 | 0.1100 | 0.0010 | 1438.9466 | `minigrid_doorkey_single_expert_latedrop_1e2_to_1e3` |
| single_expert | late_linear:0.01->0.001@0.75 | sampled_t0.7 | 0.2812 | 0.1656 | 0.4279 | 0.3670 | 0.1100 | 0.0010 | 1438.9466 | `minigrid_doorkey_single_expert_latedrop_1e2_to_1e3` |
| single_expert | late_linear:0.01->0.001@0.75 | sampled_t1.0 | 0.3750 | 0.1635 | 0.3438 | 0.4192 | 0.1100 | 0.0010 | 1438.9466 | `minigrid_doorkey_single_expert_latedrop_1e2_to_1e3` |
| single_expert | linear:0.003->0.0003 | greedy | 0.0000 | 0.0000 | 0.4955 | 0.5652 | 0.0000 | 0.0003 | 1423.6822 | `minigrid_doorkey_single_expert_linear_3e3_to_3e4` |
| single_expert | linear:0.003->0.0003 | sampled_t0.5 | 0.0000 | 0.0000 | 0.7131 | 0.5650 | 0.0000 | 0.0003 | 1423.6822 | `minigrid_doorkey_single_expert_linear_3e3_to_3e4` |
| single_expert | linear:0.003->0.0003 | sampled_t0.7 | 0.0000 | 0.0000 | 0.6057 | 0.5614 | 0.0000 | 0.0003 | 1423.6822 | `minigrid_doorkey_single_expert_linear_3e3_to_3e4` |
| single_expert | linear:0.003->0.0003 | sampled_t1.0 | 0.0000 | 0.0000 | 0.4956 | 0.5655 | 0.0000 | 0.0003 | 1423.6822 | `minigrid_doorkey_single_expert_linear_3e3_to_3e4` |
| single_expert | linear:0.01->0.001 | greedy | 0.0000 | 0.0000 | 0.5799 | 1.2175 | 0.5644 | 0.0010 | 1398.9964 | `minigrid_doorkey_single_expert_linear_1e2_to_1e3` |
| single_expert | linear:0.01->0.001 | sampled_t0.5 | 0.0000 | 0.0000 | 0.8775 | 1.2266 | 0.5644 | 0.0010 | 1398.9964 | `minigrid_doorkey_single_expert_linear_1e2_to_1e3` |
| single_expert | linear:0.01->0.001 | sampled_t0.7 | 0.1250 | 0.0466 | 0.7451 | 1.2247 | 0.5644 | 0.0010 | 1398.9964 | `minigrid_doorkey_single_expert_linear_1e2_to_1e3` |
| single_expert | linear:0.01->0.001 | sampled_t1.0 | 0.3438 | 0.1680 | 0.5831 | 1.2257 | 0.5644 | 0.0010 | 1398.9964 | `minigrid_doorkey_single_expert_linear_1e2_to_1e3` |
