# Teacher-Extraction Multi-Seed Report

This report summarizes the conditional multi-seed check for the best teacher-guided routed result:

- teacher: `flat_dense`
- student: `SARE`
- method: learner-state supervision
- evaluation path: external `policy_diagnostics` re-evaluation with `64` episodes per mode

Per-seed detail reports:

- [teacher_extraction_multiseed_seed_7.md](teacher_extraction_multiseed_seed_7.md)
- [teacher_extraction_multiseed_seed_11.md](teacher_extraction_multiseed_seed_11.md)
- [teacher_extraction_multiseed_seed_19.md](teacher_extraction_multiseed_seed_19.md)

## 64-Episode External Evaluation

| Seed | `flat_dense` Greedy | recovered `token_dense` Greedy | baseline `SARE` Greedy | learner-state `SARE` Greedy | learner-state `SARE` Best Sampled |
| --- | ---: | ---: | ---: | ---: | ---: |
| `7` | `1.0000` | `0.7031` | `0.0000` | `0.5000` | `0.5625` |
| `11` | `1.0000` | `0.0000` | `0.0000` | `1.0000` | `1.0000` |
| `19` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `0.0000` |

## Mean Success Across Seeds

| Variant | Mean Greedy Success |
| --- | ---: |
| `flat_dense` | `1.0000` |
| recovered `token_dense` | `0.5677` |
| baseline `SARE` | `0.0000` |
| learner-state `SARE` | `0.5000` |

## Interpretation

- Learner-state supervision is a real positive signal relative to baseline PPO `SARE`: it moves greedy success from `0.0000` to `0.5000` on average across the three tested seeds.
- The positive signal is not robust enough to reopen a routed greedy-performance claim:
  - seed `7` improves to `0.5000`
  - seed `11` improves to `1.0000`
  - seed `19` stays at `0.0000`
- The method therefore looks conditionally effective rather than reliably repeatable.
- Recovered `token_dense` is also seed-sensitive on this lane, but the routed result still misses a clean robustness bar because one of the three routed seeds fails completely.

## Conclusion

Teacher-guided learner-state supervision shows that `SARE` can represent a competent greedy DoorKey policy in at least some seeds, so the routed student is not ruled out on pure capacity grounds. But the current method is too brittle to support a renewed routed-performance claim in this repo.
