# Current Summary

- `flat_dense` remains the strongest greedy DoorKey control.
- recovered `token_dense` with `ppo.ent_coef=0.001` remains the canonical tokenized control.
- PPO-only `SARE` is still negative on greedy DoorKey.
- The DoorKey teacher-guided `SARE` result stays positive overall, but the final fairness-and-failure pass narrows the interpretation further:
  - on the final fresh block `47/53/59`, matched KL learner-state `single_expert` reaches mean greedy success `0.4635` versus `0.3125` for KL learner-state `SARE`, so the missing fairness control does not support thawing the claim
  - the final-block failure analysis shows high teacher confidence but higher disagreement and no better learner-state coverage on the weak `SARE` seeds, which points to extraction mismatch rather than weak teacher labels
  - the updated combined DoorKey picture still leaves KL learner-state `SARE` slightly ahead overall: `0.7122` versus `0.6862` for KL learner-state `single_expert` and `0.6354` for KL learner-state `token_dense`
  - that overall edge is too small and too block-sensitive to promote into a specifically multi-expert routed DoorKey claim
  - see [lss_final_block_single_expert_control_report.md](outputs/reports/lss_final_block_single_expert_control_report.md), [lss_final_block_failure_analysis.md](outputs/reports/lss_final_block_failure_analysis.md), [lss_frozen_claim_updated_combined_doorkey_report.md](outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.md), and [lss_frozen_claim_decision_memo.md](outputs/reports/lss_frozen_claim_decision_memo.md)
- The recovered DoorKey `SARE` policy remains causally routing-dependent under bounded eval-time probes:
  - expert ablation and fixed-router override remain strongly harmful across the expanded recovered-seed set
  - route randomization is catastrophic on most recovered seeds, but seed `29` is now a genuine narrow exception rather than a weak-probe artifact
  - see [lss_seed29_route_randomization_forensics.md](outputs/reports/lss_seed29_route_randomization_forensics.md) and [lss_broader_route_dependence_report.md](outputs/reports/lss_broader_route_dependence_report.md)
- The exact same method shows no bounded KeyCorridor transfer. See [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md).
- The current repo recommendation is to stay frozen and narrow the wording further: keep the claim explicitly teacher-guided, DoorKey-only, and external-64-episode-only, and do not strengthen it into a PPO-only, specifically multi-expert, or cross-task routed advantage. See [lss_frozen_claim_decision_memo.md](outputs/reports/lss_frozen_claim_decision_memo.md).
