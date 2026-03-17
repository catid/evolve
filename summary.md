# Current Summary

- `flat_dense` remains the strongest greedy DoorKey control.
- recovered `token_dense` with `ppo.ent_coef=0.001` remains the canonical tokenized control.
- PPO-only `SARE` is still negative on greedy DoorKey.
- The DoorKey teacher-guided `SARE` claim is now broader within DoorKey:
  - the missing matched `single_expert` control does not erase the original-lane edge: KL learner-state `single_expert` mean greedy success `0.6667`, KL learner-state `SARE` `0.7135`
  - one more fresh matched seed block stays positive for routed `SARE`: fresh-extra seeds `37/41/43` give KL learner-state `token_dense` mean `0.3333` and KL learner-state `SARE` mean `0.8229`
  - expanded nine-seed DoorKey mean: KL learner-state `token_dense` `0.5139`, KL learner-state `SARE` `0.8455`
  - no KL learner-state `SARE` DoorKey seed remains at greedy success `0.0`
  - see [lss_single_expert_matched_control_report.md](outputs/reports/lss_single_expert_matched_control_report.md), [lss_additional_fresh_seed_block_report.md](outputs/reports/lss_additional_fresh_seed_block_report.md), and [lss_expanded_combined_doorkey_report.md](outputs/reports/lss_expanded_combined_doorkey_report.md)
- The recovered DoorKey `SARE` policy remains causally routing-dependent under bounded eval-time probes beyond the original 2-seed demo:
  - expert ablation and fixed-router override still collapse or severely damage recovered success on seeds `7`, `19`, `23`, and `29`
  - route randomization is still catastrophic on `7`, `19`, and `23`, but only weakly harmful on `29`
  - see [lss_extended_route_dependence_report.md](outputs/reports/lss_extended_route_dependence_report.md)
- The exact same method shows no bounded KeyCorridor transfer. See [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md).
- The current repo recommendation is to continue within DoorKey only: keep the claim explicitly teacher-guided and DoorKey-only, and do not generalize it into a PPO-only or cross-task routed advantage. See [lss_claim_broadening_decision_memo.md](outputs/reports/lss_claim_broadening_decision_memo.md).
