# Current Summary

- `flat_dense` remains the strongest greedy DoorKey control.
- recovered `token_dense` with `ppo.ent_coef=0.001` remains the canonical tokenized control.
- PPO-only `SARE` is still negative on greedy DoorKey.
- The DoorKey teacher-guided `SARE` claim stays positive but is no longer strong enough to promote into a specifically multi-expert routed edge:
  - on the matched structured DoorKey slice, KL learner-state `SARE` still leads: `0.8455` vs `0.7604` for KL learner-state `single_expert` and `0.5139` for KL learner-state `token_dense`
  - one final fresh matched seed block weakens that edge materially: final fresh seeds `47/53/59` give KL learner-state `token_dense` mean `1.0000` and KL learner-state `SARE` mean `0.3125`
  - the full combined DoorKey picture still favors KL learner-state `SARE` over KL learner-state `token_dense`, but not clearly enough over `single_expert` to call it specifically multi-expert
  - see [lss_fresh_single_expert_matched_control_report.md](outputs/reports/lss_fresh_single_expert_matched_control_report.md), [lss_final_fresh_seed_block_report.md](outputs/reports/lss_final_fresh_seed_block_report.md), and [lss_final_combined_doorkey_report.md](outputs/reports/lss_final_combined_doorkey_report.md)
- The recovered DoorKey `SARE` policy remains causally routing-dependent under bounded eval-time probes:
  - expert ablation and fixed-router override remain strongly harmful across the expanded recovered-seed set
  - route randomization is catastrophic on most recovered seeds, but seed `29` is now a genuine narrow exception rather than a weak-probe artifact
  - see [lss_seed29_route_randomization_forensics.md](outputs/reports/lss_seed29_route_randomization_forensics.md) and [lss_broader_route_dependence_report.md](outputs/reports/lss_broader_route_dependence_report.md)
- The exact same method shows no bounded KeyCorridor transfer. See [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md).
- The current repo recommendation is to stay frozen: keep the claim explicitly teacher-guided and DoorKey-only, and do not strengthen it into a PPO-only, specifically multi-expert, or cross-task routed advantage. See [lss_multi_expert_hardening_decision_memo.md](outputs/reports/lss_multi_expert_hardening_decision_memo.md).
