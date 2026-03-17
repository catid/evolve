# Current Summary

- `flat_dense` remains the strongest greedy DoorKey control.
- recovered `token_dense` with `ppo.ent_coef=0.001` remains the canonical tokenized control.
- PPO-only `SARE` is still negative on greedy DoorKey.
- The DoorKey teacher-guided `SARE` claim is now stronger under fresh matched controls:
  - fresh seeds `23/29/31`: KL learner-state `token_dense` reaches `0.0000/0.6250/1.0000`, while KL learner-state `SARE` stays at `1.0000/1.0000/1.0000`
  - combined six-seed DoorKey mean: KL learner-state `token_dense` `0.6042`, KL learner-state `SARE` `0.8568`
  - no KL learner-state `SARE` seed remains at greedy success `0.0`
  - see [lss_fresh_matched_control_report.md](outputs/reports/lss_fresh_matched_control_report.md) and [lss_combined_doorkey_report.md](outputs/reports/lss_combined_doorkey_report.md)
- The recovered DoorKey `SARE` policy is now causally routing-dependent under bounded eval-time probes: on recovered seeds `7` and `23`, every single-expert ablation, fixed-router override, and route randomization drops greedy success from `1.0` to `0.0`. See [lss_causal_route_dependence_report.md](outputs/reports/lss_causal_route_dependence_report.md).
- The exact same method shows no bounded KeyCorridor transfer. See [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md).
- The current repo recommendation is to broaden within DoorKey only: keep the claim explicitly teacher-guided and DoorKey-only, and do not generalize it into a PPO-only or cross-task routed advantage. See [lss_claim_consolidation_decision_memo.md](outputs/reports/lss_claim_consolidation_decision_memo.md).
