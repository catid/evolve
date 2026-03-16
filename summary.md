# Current Summary

- `flat_dense` remains the strongest greedy DoorKey control.
- recovered `token_dense` with `ppo.ent_coef=0.001` remains the canonical tokenized control.
- PPO-only `SARE` is still negative on greedy DoorKey.
- The reopened teacher-guided DoorKey `SARE` claim is now stronger on fresh seeds: KL learner-state `SARE` reached greedy `1.0` on all added DoorKey seeds `23`, `29`, and `31`. See [lss_additional_seed_report.md](outputs/reports/lss_additional_seed_report.md).
- Matched teacher-guided KL learner-state supervision helps `token_dense` too, but `SARE` still keeps the higher DoorKey mean on the original `7/11/19` lane. See [lss_matched_control_report.md](outputs/reports/lss_matched_control_report.md).
- On the newly recovered DoorKey seed `23`, routing still looks meaningful rather than collapsed. See [lss_new_case_route_integrity_report.md](outputs/reports/lss_new_case_route_integrity_report.md).
- The exact same method shows no bounded KeyCorridor transfer. See [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md).
- The current repo recommendation is to stay narrowly scoped: keep the DoorKey teacher-guided extraction claim, but do not generalize it into a PPO-only or cross-task routed advantage. See [lss_claim_hardening_decision_memo.md](outputs/reports/lss_claim_hardening_decision_memo.md).
