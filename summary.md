# Current Summary

- `flat_dense` remains the strongest verified greedy DoorKey control.
- `token_dense` with `ppo.ent_coef=0.001` remains the canonical recovered tokenized DoorKey control.
- The offline teacher-distillation path was negative for both tokenized and routed students: it improved sampled behavior but never recovered greedy DoorKey behavior. See [policy_distillation_report.md](outputs/reports/policy_distillation_report.md).
- Learner-state supervision from a `flat_dense` teacher partially recovered greedy `SARE`, reaching seed-level greedy success `0.5`, `1.0`, and `0.0` across the 3-seed check. See [teacher_extraction_multiseed_report.md](outputs/reports/teacher_extraction_multiseed_report.md).
- The strongest recovered `SARE` seed retained meaningful routing rather than collapsing into an obvious dense shortcut. See [distilled_route_integrity_best_seed_report.md](outputs/reports/distilled_route_integrity_best_seed_report.md).
- Current recommendation: pause routed greedy-performance claims on DoorKey. The teacher-guided `SARE` gain is real but not robust enough to reopen a routed win. See [teacher_extraction_decision_memo.md](outputs/reports/teacher_extraction_decision_memo.md).
