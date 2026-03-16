# Distilled Route Integrity Report

- episodes: `64`

| Run | Variant | Eval Success | Eval Return | Route Entropy | Path Entropy | Active Compute | expert_load_0 | expert_load_1 | expert_load_2 | expert_load_3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sare_ent1e3 | sare | 0.0000 | 0.0000 | 1.3832 | 0.9307 | 0.5000 | 0.2381 | 0.2494 | 0.2454 | 0.2670 |
| flat_dense_to_sare_lss | sare | 1.0000 | 0.9634 | 1.3383 | 1.2850 | 0.5000 | 0.2708 | 0.2434 | 0.2383 | 0.2475 |

## Interpretation

- `flat_dense_to_sare_lss` changes greedy success from `0.0000` to `1.0000`.
- Route entropy moves from `1.3832` to `1.3383`, and active compute proxy moves from `0.5000` to `0.5000`.
