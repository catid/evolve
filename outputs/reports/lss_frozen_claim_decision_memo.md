# Frozen-Claim Decision Memo

## Decision

The right final claim is: **narrower method-first result**.

## Answers

1. On the final fresh block, KL learner-state SARE mean greedy success is `0.3125` versus `1.0000` for KL learner-state token_dense and `0.4635` for KL learner-state single_expert. See [lss_final_block_single_expert_control_report.md](lss_final_block_single_expert_control_report.md).
2. The most plausible explanation for the `47/53/59` reversal comes from the failure analysis report: teacher labels stay confident, but the weak seeds show extraction mismatch and route-specific fragility on the final block rather than a clean teacher-quality problem. See [lss_final_block_failure_analysis.md](lss_final_block_failure_analysis.md).
3. Across the updated combined DoorKey picture, mean greedy success is `0.6354` for KL learner-state token_dense, `0.6862` for KL learner-state single_expert, and `0.7122` for KL learner-state SARE. See [lss_frozen_claim_updated_combined_doorkey_report.md](lss_frozen_claim_updated_combined_doorkey_report.md).

Recommendation: narrow further. The final-block fairness control and failure analysis make the positive result look more like a structured-student extraction win than a specifically routed advantage.
