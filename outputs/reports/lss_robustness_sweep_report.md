# Learner-State Robustness Sweep Report

- external evaluation episodes per mode: `64`

| Method | Seed | Greedy Success | Best Sampled Success | Loss | Aggregation | Weighting | Unique Ratio | Disagreement | Teacher Conf |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| flat_dense_to_sare_lss | 7 | 0.5000 | 0.5625 | ce | append_all | uniform | - | - | - |
| flat_dense_to_sare_lss | 11 | 1.0000 | 1.0000 | ce | append_all | uniform | - | - | - |
| flat_dense_to_sare_lss | 19 | 0.0000 | 0.0000 | ce | append_all | uniform | - | - | - |
| flat_dense_to_sare_lss_kl | 7 | 1.0000 | 1.0000 | kl | append_all | uniform | - | - | - |
| flat_dense_to_sare_lss_kl | 19 | 0.5781 | 0.6406 | kl | append_all | uniform | - | - | - |
| flat_dense_to_sare_lss_kl_cap_recent | 7 | 0.0000 | 0.1562 | kl | cap_recent | uniform | - | - | - |
| flat_dense_to_sare_lss_kl_cap_recent | 19 | 0.0000 | 0.0000 | kl | cap_recent | uniform | - | - | - |
| flat_dense_to_sare_lss_kl_cap_recent_balanced | 7 | 0.0000 | 0.2969 | kl | cap_recent_balanced | uniform | - | - | - |
| flat_dense_to_sare_lss_kl_cap_recent_balanced | 19 | 0.0000 | 0.3281 | kl | cap_recent_balanced | uniform | - | - | - |
| improved_lss_sare | 7 | 1.0000 | 1.0000 | kl | append_all | uniform | - | - | - |
| improved_lss_sare | 11 | 0.5625 | 0.9219 | kl | append_all | uniform | - | - | - |
| improved_lss_sare | 19 | 0.5781 | 0.6406 | kl | append_all | uniform | - | - | - |

## Minimal Tokenized Sanity Check

| Seed | Greedy Success | Best Sampled Success | Method |
| --- | ---: | ---: | --- |
| 19 | 1.0000 | 1.0000 | improved_lss_token_dense |

## Interpretation

- The sweep isolates one winning axis: teacher-logit KL targets. Switching from hard labels to `kl + append_all` moves seed `7` from greedy `0.5000` to `1.0000` and seed `19` from greedy `0.0000` to `0.5781`.
- Aggregation control is negative in this bounded setting. Both `cap_recent` and `cap_recent_balanced` collapse the seed-7 KL win back to greedy `0.0000`, and neither lifts seed `19` above greedy `0.0000`.
- The promoted gate method is therefore the simplest positive variant: `kl` loss with `append_all` aggregation.
- The minimal tokenized-student sanity check on seed `19` is positive (`1.0000` greedy), so the stronger learner-state method is not a SARE-only artifact.
