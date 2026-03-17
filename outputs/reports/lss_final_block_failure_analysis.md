# Final-Block Failure Analysis

- external evaluation episodes per mode: `64`

## Final-Block Fairness Context

| Seed | KL learner-state token_dense | KL learner-state single_expert | KL learner-state SARE |
| --- | ---: | ---: | ---: |
| 47 | 1.0 | 0.453125 | 0.0 |
| 53 | 1.0 | 0.515625 | 0.515625 |
| 59 | 1.0 | 0.421875 | 0.421875 |

## Best-Round Learner-State Diagnostics

| Seed | Variant | Best Round | Best Greedy | Final Greedy | Disagreement | Teacher Conf | Unique Ratio | Added Steps | Aggregate Steps | Student Conf | Route Entropy | Path Entropy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 47 | KL learner-state SARE | 1 | 0.0000 | 0.0000 | 1.0000 | 0.9911 | 0.0016 | 16000.0000 | 16000.0000 | 0.3272 | 1.3846 | 0.5627 |
| 53 | KL learner-state SARE | 3 | 0.5156 | 0.5156 | 0.3731 | 0.9582 | 0.0029 | 16000.0000 | 48000.0000 | 0.9460 | 1.3770 | 0.7234 |
| 59 | KL learner-state SARE | 4 | 0.4219 | 0.4219 | 0.9715 | 0.9992 | 0.0032 | 16000.0000 | 64000.0000 | 0.8045 | 1.3833 | 1.0304 |
| 47 | KL learner-state single_expert | 4 | 0.4531 | 0.4531 | 0.6307 | 0.9535 | 0.0035 | 16000.0000 | 64000.0000 | 0.9065 | 0.0000 | 0.0000 |
| 53 | KL learner-state single_expert | 3 | 0.5156 | 0.5156 | 0.3731 | 0.9582 | 0.0029 | 16000.0000 | 48000.0000 | 0.9278 | 0.0000 | 0.0000 |
| 59 | KL learner-state single_expert | 4 | 0.4219 | 0.4219 | 0.9752 | 0.9969 | 0.0032 | 16000.0000 | 64000.0000 | 0.9020 | 0.0000 | 0.0000 |

## State-Local Route Redundancy Proxy

| Seed | Mean Unique Pairs / Obs | Mean Dominant Pair Fraction | Global Top Pair Fraction |
| --- | ---: | ---: | ---: |
| 47 | 3.8281 | 0.7128 | 0.7128 |
| 53 | 4.0000 | 0.5852 | 0.5852 |
| 59 | 4.2920 | 0.4548 | 0.4548 |

## Final-Block Route Probes

| Seed | Baseline | Worst Expert Ablation | Fixed Router | Route Randomization |
| --- | ---: | ---: | ---: | ---: |
| 47 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 53 | 0.5156 | 0.0000 | 0.0000 | 0.0000 |
| 59 | 0.4219 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation

- The final fresh block weakness is not specifically multi-expert: matched KL learner-state single_expert matches or beats KL learner-state SARE on the same seeds.
- Relative to the stronger recovered seeds, the final block shows higher best-round teacher-student disagreement (`0.7815` vs `0.5811`) without better learner-state coverage (`0.0026` vs `0.0038`), which points to extraction mismatch rather than weak teacher labels.
- Even on the weak final-block seeds, fixed-router override still collapses success, so routing remains causally used rather than obviously bypassed.
- Because matched single_expert is at least as strong on the final block, the failure pattern supports a method-first claim more than a specifically routed interpretation.
