# Distilled Route Integrity Report

- episodes: `64`

| Run | Variant | Eval Success | Eval Return | Route Entropy | Path Entropy | Active Compute | expert_load_0 | expert_load_1 | expert_load_2 | expert_load_3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sare_ent1e3 | sare | 0.0000 | 0.0000 | 1.3857 | 0.3064 | 0.5000 | 0.2458 | 0.2453 | 0.2632 | 0.2456 |
| kl_lss_sare | sare | 1.0000 | 0.9651 | 1.3804 | 0.8050 | 0.5000 | 0.2438 | 0.2489 | 0.2252 | 0.2821 |
| improved_lss_sare | sare | 0.5938 | 0.5754 | 1.3837 | 0.9446 | 0.5000 | 0.2463 | 0.2643 | 0.2485 | 0.2409 |

## Interpretation

- `kl_lss_sare` changes greedy success from `0.0000` to `1.0000`.
- Route entropy moves from `1.3857` to `1.3804`, and active compute proxy moves from `0.5000` to `0.5000`.
