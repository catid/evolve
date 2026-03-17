# Resume-Gate Decision Memo

## Decision

The right final state is: **stay frozen as-is**.

## Answers

1. Actionable mechanism status: `mechanism plausible but weak`. See [lss_resume_gate_failure_mechanism_report.md](lss_resume_gate_failure_mechanism_report.md).
2. Bounded resume attempt justified: `no`. The final block is `0.3125` for KL learner-state `SARE`, `0.4635` for KL learner-state `single_expert`, and `1.0000` for KL learner-state `token_dense`, so the missing fairness control does not earn a retry. See [lss_final_block_single_expert_control_report.md](lss_final_block_single_expert_control_report.md).
3. Resume attempt run: `no`. No preregistered plan was written because the mechanism audit did not clear the resume gate. See [lss_resume_gate_failure_mechanism_report.md](lss_resume_gate_failure_mechanism_report.md).
4. The combined DoorKey picture remains `0.7122` for KL learner-state `SARE`, `0.6862` for KL learner-state `single_expert`, and `0.6354` for KL learner-state `token_dense`; that is still not strong enough to thaw the claim. See [lss_frozen_claim_updated_combined_doorkey_report.md](lss_frozen_claim_updated_combined_doorkey_report.md).

Recommendation: stay frozen. The current DoorKey teacher-guided `SARE` result remains a bounded positive result, but this resume gate does not justify a new retry.
