# Current Summary

## Frozen Claim Scope

- Allowed: bounded teacher-guided DoorKey `SARE` result only.
- Not allowed: PPO-only routed win, specifically multi-expert routed advantage, cross-task routed advantage, or KeyCorridor transfer claim.
- Any future thaw candidate must beat retry-block KL learner-state `SARE` mean `0.3125` on seeds `47/53/59`, at least match same-block KL learner-state `single_expert`, and preserve combined DoorKey KL learner-state `SARE` mean `0.7122`.
- Canonical gate artifacts: [frozen_claim_envelope.md](outputs/reports/frozen_claim_envelope.md), [frozen_claim_manifest_report.md](outputs/reports/frozen_claim_manifest_report.md), [frozen_baseline_validation.md](outputs/reports/frozen_baseline_validation.md), and [claim_gate_dry_run.md](outputs/reports/claim_gate_dry_run.md).
- Frozen benchmark pack artifacts: [frozen_benchmark_pack.md](outputs/reports/frozen_benchmark_pack.md), [frozen_benchmark_pack_validation.md](outputs/reports/frozen_benchmark_pack_validation.md), [claim_gate_pack_dry_run.md](outputs/reports/claim_gate_pack_dry_run.md), and [freeze_hardening_operational_memo.md](outputs/reports/freeze_hardening_operational_memo.md).

- `flat_dense` remains the strongest greedy DoorKey control.
- recovered `token_dense` with `ppo.ent_coef=0.001` remains the canonical tokenized control.
- PPO-only `SARE` is still negative on greedy DoorKey.
- The DoorKey teacher-guided `SARE` result stays positive overall, and the deeper forensic atlas still keeps it frozen:
  - on the final fresh block `47/53/59`, matched KL learner-state `single_expert` reaches mean greedy success `0.4635` versus `0.3125` for KL learner-state `SARE`, so the missing fairness control does not support thawing the claim
  - the trajectory casebook and route-locality pass show a mixed mechanism split: seed `47` is the clearest route-fragile `SARE` failure, while `53/59` look more like shared structured-student post-unlock collapse
  - the round audit and scorecard still end at `bounded retry not justified`: final-block seeds show high teacher confidence, higher disagreement, lower coverage, and elevated route-pair concentration, but not one clean lever that earns a bounded retry
  - the updated combined DoorKey picture still leaves KL learner-state `SARE` slightly ahead overall: `0.7122` versus `0.6862` for KL learner-state `single_expert` and `0.6354` for KL learner-state `token_dense`
  - that overall edge is still too small and too block-sensitive to justify a resume attempt or a specifically multi-expert routed DoorKey claim
  - see [lss_forensic_casebook.md](outputs/reports/lss_forensic_casebook.md), [lss_forensic_round_audit.md](outputs/reports/lss_forensic_round_audit.md), [lss_forensic_route_locality.md](outputs/reports/lss_forensic_route_locality.md), [lss_resume_qualification_scorecard.md](outputs/reports/lss_resume_qualification_scorecard.md), and [lss_forensic_atlas_decision_memo.md](outputs/reports/lss_forensic_atlas_decision_memo.md)
- The recovered DoorKey `SARE` policy remains causally routing-dependent under bounded eval-time probes:
  - expert ablation and fixed-router override remain strongly harmful across the expanded recovered-seed set
  - route randomization is catastrophic on most recovered seeds, but seed `29` is now a genuine narrow exception rather than a weak-probe artifact
  - see [lss_seed29_route_randomization_forensics.md](outputs/reports/lss_seed29_route_randomization_forensics.md) and [lss_broader_route_dependence_report.md](outputs/reports/lss_broader_route_dependence_report.md)
- The exact same method shows no bounded KeyCorridor transfer. See [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md).
- The current repo recommendation is to stay frozen as-is and require the automated gate before any thaw consideration. See [freeze_hardening_decision_memo.md](outputs/reports/freeze_hardening_decision_memo.md).

## Claim Gate

- Validate the sealed frozen benchmark pack first: [frozen_benchmark_pack_validation.md](outputs/reports/frozen_benchmark_pack_validation.md)
- Package any future thaw candidate using: [candidate_result_pack_schema.md](outputs/reports/candidate_result_pack_schema.md) and [candidate_result_pack_template.json](outputs/reports/candidate_result_pack_template.json)
- The current frozen candidate still fails the pack-based gate: [claim_gate_pack_dry_run.md](outputs/reports/claim_gate_pack_dry_run.md)
- Malformed or incomplete candidates should be treated as missing prerequisites: [claim_gate_pack_inconclusive.md](outputs/reports/claim_gate_pack_inconclusive.md)
- The adversarial corpus now exercises PASS / FAIL / INCONCLUSIVE paths and is the conformance reference for future gate changes: [claim_gate_corpus_report.md](outputs/reports/claim_gate_corpus_report.md) and [claim_gate_conformance_report.md](outputs/reports/claim_gate_conformance_report.md)
- Ad hoc markdown reports are insufficient for thaw consideration; the candidate pack plus pack-based gate are the only thaw triage surface. See [claim_gate_redteam_decision_memo.md](outputs/reports/claim_gate_redteam_decision_memo.md).
- The gate is now also replay-checked against the repo's real historical claim phases, not just synthetic red-team cases. See [historical_candidate_pack_catalog.md](outputs/reports/historical_candidate_pack_catalog.md), [claim_history_replay_report.md](outputs/reports/claim_history_replay_report.md), [claim_ledger_consistency_audit.md](outputs/reports/claim_ledger_consistency_audit.md), and [claim_history_replay_decision_memo.md](outputs/reports/claim_history_replay_decision_memo.md).
