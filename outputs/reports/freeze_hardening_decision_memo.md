# Freeze-Hardening Decision Memo

## Answers

1. Current allowed claim envelope: `bounded teacher-guided DoorKey SARE result` only. Not allowed: `ppo_only_routed_win, specifically_multi_expert_routed_advantage, cross_task_routed_advantage, keycorridor_transfer_claim`. See [frozen_claim_envelope.md](frozen_claim_envelope.md).
2. The frozen state is defined by the manifest `doorkey_frozen_claim` plus the authoritative combined and retry-block reports. The key thresholds are combined KL learner-state `SARE` mean `0.7122` and retry-block KL learner-state `SARE` mean `0.3125`. See [frozen_claim_manifest_report.md](frozen_claim_manifest_report.md).
3. Any future thaw candidate must use the external 64-episode DoorKey path, include the required fairness controls, beat the frozen retry-block KL learner-state `SARE` mean `0.3125`, at least match same-block KL learner-state `single_expert`, and preserve the combined DoorKey KL learner-state `SARE` mean `0.7122`. See [future_retry_template.md](future_retry_template.md).
4. The project is now set up to resist overclaiming: the frozen baseline validation currently returns `PASS: frozen baseline validated`, and the automated dry-run claim gate returns `FAIL: claim remains frozen` on the current frozen artifacts. See [frozen_baseline_validation.md](frozen_baseline_validation.md) and [claim_gate_dry_run.md](claim_gate_dry_run.md).
5. Recommendation: stay frozen until a preregistered retry clears the automated gate.
