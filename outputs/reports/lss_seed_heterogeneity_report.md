# Learner-State Seed Heterogeneity Report

- external evaluation episodes per mode: `64`

| Seed | Round | Greedy Success | Best Sampled Success | Added Steps | Unique Ratio | Teacher Conf | Disagreement | Collection Route Entropy | Eval Route Entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0 | 0.0000 | 1.0000 | - | - | - | - | - | 1.3855 |
| 7 | 1 | 0.0000 | 0.0000 | 16000.0000 | 0.0019 | 0.9821 | 0.9980 | 1.3855 | 1.3850 |
| 7 | 2 | 0.0000 | 0.0000 | 16000.0000 | 0.0026 | 0.9843 | 0.9936 | 1.3850 | 1.3846 |
| 7 | 3 | 0.0000 | 0.9375 | 16000.0000 | 0.0027 | 0.9929 | 0.2117 | 1.3846 | 1.3837 |
| 7 | 4 | 0.5000 | 0.5625 | 16000.0000 | 0.0030 | 0.9984 | 0.9763 | 1.3836 | 1.3821 |
| 11 | 0 | 0.0000 | 0.7812 | - | - | - | - | - | 1.3832 |
| 11 | 1 | 0.0000 | 0.0000 | 16000.0000 | 0.0019 | 0.9935 | 0.9998 | 1.3832 | 1.3646 |
| 11 | 2 | 0.0000 | 0.0312 | 16000.0000 | 0.0026 | 0.9891 | 0.9394 | 1.3643 | 1.3566 |
| 11 | 3 | 0.5625 | 0.7344 | 16000.0000 | 0.0028 | 0.9979 | 0.4932 | 1.3567 | 1.3467 |
| 11 | 4 | 1.0000 | 1.0000 | 6108.0000 | 0.0087 | 0.9984 | 0.9037 | 1.3467 | 1.3384 |
| 19 | 0 | 0.0000 | 0.9688 | - | - | - | - | - | 1.3823 |
| 19 | 1 | 0.0000 | 0.0000 | 16000.0000 | 0.0018 | 0.9597 | 0.9996 | 1.3821 | 1.3819 |
| 19 | 2 | 0.0000 | 0.0156 | 16000.0000 | 0.0026 | 0.9940 | 0.9943 | 1.3819 | 1.3834 |
| 19 | 3 | 0.0000 | 0.2031 | 16000.0000 | 0.0029 | 0.9964 | 0.4926 | 1.3834 | 1.3822 |
| 19 | 4 | 0.0000 | 0.0000 | 16000.0000 | 0.0032 | 0.9951 | 0.9785 | 1.3822 | 1.3830 |

## Interpretation

- Seed `11` is the successful case: greedy success stays at `0.0` until round `3`, then reaches `0.5625` at round `3` and `1.0000` at round `4`.
- Seed `7` is partial: it stays at `0.0` through round `3` and only reaches `0.5000` at round `4`.
- Seed `19` is the failed case for the original hard-label learner-state method: its best sampled round is round `3` (`0.2031`), but greedy success never rises above `0.0`.
- Teacher confidence is not the main differentiator. By rounds `3` and `4`, all three seeds are already seeing high-confidence teacher labels (`0.9929` to `0.9984` on seed `7`, `0.9979` to `0.9984` on seed `11`, and `0.9951` to `0.9964` on seed `19`).
- The more actionable difference is dataset dynamics. The successful seed `11` is the only one whose winning round shrinks the labeled batch sharply (`6108` steps instead of `16000`) and raises unique-state ratio from about `0.0028` to `0.0087`, while the failed seed `19` keeps training on the full append-all dataset with unique-state ratio stuck near `0.003`.
- Seed `19` round `3` already matches seed `11` round `3` on teacher confidence and disagreement closely enough that no single scalar fully explains the brittleness. The most defensible diagnosis is an optimization/aggregation issue: weak teacher confidence does not explain the failure, and stale append-all data is a more plausible suspect than low-quality labels.
