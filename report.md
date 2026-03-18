# DoorKey Extraction Report

## Current Conclusion

- `flat_dense` remains the strongest greedy DoorKey control.
- recovered `token_dense` (`ppo.ent_coef=0.001`) remains the canonical tokenized control.
- PPO-only `SARE` still loses the fair greedy DoorKey comparison.
- The frozen benchmark pack remains the baseline comparison unit for DoorKey thaw work.
- The first staged candidate that clears that baseline now exists inside the same teacher-guided `KL` learner-state family:
  - candidate: `post_unlock_weighted`
  - scope: DoorKey only
  - evaluation: external `64`-episode `policy_diagnostics`
  - result: `PASS: thaw consideration allowed`
- This is still not a PPO-only, specifically multi-expert, cross-task, or KeyCorridor claim.

## Frozen Benchmark Baseline

Allowed:

- bounded teacher-guided DoorKey `SARE` result only

Not allowed:

- PPO-only routed win
- specifically multi-expert routed DoorKey advantage
- cross-task routed advantage
- KeyCorridor transfer claim

Future thaw candidates must clear all of:

- external `64`-episode `policy_diagnostics` evaluation
- retry-block KL learner-state `SARE` mean strictly above `0.3125` on seeds `47/53/59`
- retry-block KL learner-state `SARE` at least matching same-block KL learner-state `single_expert`
- combined DoorKey KL learner-state `SARE` mean preserved at or above `0.7122`

Canonical gate artifacts:

- [frozen_claim_envelope.md](outputs/reports/frozen_claim_envelope.md)
- [frozen_claim_manifest_report.md](outputs/reports/frozen_claim_manifest_report.md)
- [frozen_baseline_validation.md](outputs/reports/frozen_baseline_validation.md)
- [claim_gate_dry_run.md](outputs/reports/claim_gate_dry_run.md)
- [freeze_hardening_decision_memo.md](outputs/reports/freeze_hardening_decision_memo.md)

## Frozen Benchmark Pack

The frozen DoorKey state remains sealed as an explicit benchmark pack instead of only a report bundle.

- [frozen_benchmark_pack.md](outputs/reports/frozen_benchmark_pack.md)
- [frozen_benchmark_pack.json](outputs/reports/frozen_benchmark_pack.json)
- [frozen_benchmark_pack_validation.md](outputs/reports/frozen_benchmark_pack_validation.md)
- [benchmark_pack_schema_report.md](outputs/reports/benchmark_pack_schema_report.md)

## Claim Gate

Future thaw discussion must use the pack-based gate, not ad hoc report comparison.

- candidate pack format: [candidate_result_pack_schema.md](outputs/reports/candidate_result_pack_schema.md)
- candidate pack template: [candidate_result_pack_template.json](outputs/reports/candidate_result_pack_template.json)
- current frozen dry run: [claim_gate_pack_dry_run.md](outputs/reports/claim_gate_pack_dry_run.md)
- long-campaign passing candidate: [long_campaign_candidate_pack.json](outputs/reports/long_campaign_candidate_pack.json)
- long-campaign gate result: [long_campaign_gate_report.md](outputs/reports/long_campaign_gate_report.md)
- malformed candidate example: [claim_gate_pack_inconclusive.md](outputs/reports/claim_gate_pack_inconclusive.md)
- adversarial conformance corpus: [claim_gate_corpus_report.md](outputs/reports/claim_gate_corpus_report.md)
- adversarial conformance result: [claim_gate_conformance_report.md](outputs/reports/claim_gate_conformance_report.md)
- red-team memo: [claim_gate_redteam_decision_memo.md](outputs/reports/claim_gate_redteam_decision_memo.md)
- historical replay catalog: [historical_candidate_pack_catalog.md](outputs/reports/historical_candidate_pack_catalog.md)
- historical replay report: [claim_history_replay_report.md](outputs/reports/claim_history_replay_report.md)
- historical replay audit: [claim_ledger_consistency_audit.md](outputs/reports/claim_ledger_consistency_audit.md)
- historical replay memo: [claim_history_replay_decision_memo.md](outputs/reports/claim_history_replay_decision_memo.md)
- operational memo: [freeze_hardening_operational_memo.md](outputs/reports/freeze_hardening_operational_memo.md)

Ad hoc markdown reports are insufficient for thaw consideration. Any future DoorKey thaw candidate must be packaged as a candidate result pack and clear the pack-based gate.
The hardened trust path is now:

- sealed frozen benchmark pack
- pack-based claim gate
- adversarial conformance corpus
- historical replay against the repo's real claim history

## Final Decision Path

Source artifacts:

- [frozen_claim_envelope.md](outputs/reports/frozen_claim_envelope.md)
- [frozen_claim_manifest_report.md](outputs/reports/frozen_claim_manifest_report.md)
- [frozen_baseline_validation.md](outputs/reports/frozen_baseline_validation.md)
- [claim_gate_dry_run.md](outputs/reports/claim_gate_dry_run.md)
- [claim_ledger.md](outputs/reports/claim_ledger.md)
- [future_retry_template.md](outputs/reports/future_retry_template.md)
- [freeze_hardening_decision_memo.md](outputs/reports/freeze_hardening_decision_memo.md)
- [lss_forensic_atlas_reproduction_note.md](outputs/reports/lss_forensic_atlas_reproduction_note.md)
- [lss_forensic_casebook.md](outputs/reports/lss_forensic_casebook.md)
- [lss_forensic_round_audit.md](outputs/reports/lss_forensic_round_audit.md)
- [lss_forensic_route_locality.md](outputs/reports/lss_forensic_route_locality.md)
- [lss_resume_qualification_scorecard.md](outputs/reports/lss_resume_qualification_scorecard.md)
- [lss_forensic_atlas_decision_memo.md](outputs/reports/lss_forensic_atlas_decision_memo.md)
- [lss_final_block_single_expert_control_report.md](outputs/reports/lss_final_block_single_expert_control_report.md)
- [lss_frozen_claim_updated_combined_doorkey_report.md](outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.md)
- [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md)
- [long_campaign_registration.md](outputs/reports/long_campaign_registration.md)
- [long_campaign_mechanism_shortlist.md](outputs/reports/long_campaign_mechanism_shortlist.md)
- [long_campaign_stage2_screening.md](outputs/reports/long_campaign_stage2_screening.md)
- [long_campaign_stage3_fairness.md](outputs/reports/long_campaign_stage3_fairness.md)
- [long_campaign_stage4_replication.md](outputs/reports/long_campaign_stage4_replication.md)
- [long_campaign_stage5_route_validation.md](outputs/reports/long_campaign_stage5_route_validation.md)
- [long_campaign_gate_report.md](outputs/reports/long_campaign_gate_report.md)
- [long_campaign_decision_memo.md](outputs/reports/long_campaign_decision_memo.md)

All final claims in this phase use the external `64`-episode `policy_diagnostics` path.

## Long Campaign Result

The staged DoorKey-only campaign produced the first within-envelope thaw-qualified candidate.

- Stage 2 weak-block screening kept only `post_unlock_weighted`; all recency and phase-balanced variants collapsed to `0.0000` mean on `47/53/59`. See [long_campaign_stage2_screening.md](outputs/reports/long_campaign_stage2_screening.md).
- Stage 3 fairness showed the candidate raises weak-block KL learner-state `SARE` mean from frozen `0.3125` to `0.4635`, exactly matching same-block KL learner-state `single_expert`, while KL learner-state `token_dense` stays at `1.0000`. See [long_campaign_stage3_fairness.md](outputs/reports/long_campaign_stage3_fairness.md).
- Stage 4 replication preserved and improved the stronger historical picture:
  - combined KL learner-state `SARE`: `0.7500`
  - combined KL learner-state `single_expert`: `0.6862`
  - combined KL learner-state `token_dense`: `0.6354`
  - no complete-seed failures for candidate KL learner-state `SARE`
  - no new complete-seed failures on previously healthy strong seeds
  See [long_campaign_stage4_replication.md](outputs/reports/long_campaign_stage4_replication.md).
- Stage 5 route checks still show materially harmful routing disruption on the selected weak case `(fresh_final, 53)` and strong case `(fresh, 23)`. See [long_campaign_stage5_route_validation.md](outputs/reports/long_campaign_stage5_route_validation.md).
- Stage 6 packaged the candidate and the automated gate returned `PASS: thaw consideration allowed`. See [long_campaign_gate_report.md](outputs/reports/long_campaign_gate_report.md) and [long_campaign_decision_memo.md](outputs/reports/long_campaign_decision_memo.md).

The right current wording is now:

- thaw consideration is allowed within DoorKey only
- still teacher-guided only
- still KL learner-state only
- still external `64`-episode evaluation only
- still not a PPO-only, specifically multi-expert, cross-task, or KeyCorridor claim

## What Still Stands Negative

Source artifacts:

- [checkpoint_dynamics_report.md](outputs/reports/checkpoint_dynamics_report.md)
- [entropy_schedule_report.md](outputs/reports/entropy_schedule_report.md)
- [self_imitation_report.md](outputs/reports/self_imitation_report.md)
- [policy_distillation_report.md](outputs/reports/policy_distillation_report.md)

These earlier no-go results still stand:

- checkpoint selection did not reveal a good greedy PPO `SARE` policy
- entropy schedules did not recover greedy PPO `SARE`
- self-imitation did not recover greedy PPO `SARE`
- offline teacher distillation did not recover greedy `SARE`

So the repo’s positive routed result still does not come from PPO tuning or offline imitation.

## What Changed In The Forensic-Atlas Phase

The forensic-atlas phase answered the last DoorKey-only question:

- is there a specific, auditable failure mechanism behind the weak `47/53/59` block that justifies one bounded retry?

The answer is:

- the deeper trajectory casebook, round audit, and phase-local route analysis show a mixed mechanism story rather than one clean failure mode
- seed `47` is the clearest route-fragile `SARE` case, while seeds `53` and `59` look closer to shared structured-student post-unlock collapse
- the weak block still has only a plausible but weak mechanism signature overall
- that signature is not clean or actionable enough to justify a bounded retry
- no on thawing the claim
- still no on bounded KeyCorridor transfer

On the matched structured DoorKey slice where all three structured students exist:

| Lane | Seed | recovered `token_dense` | KL learner-state `token_dense` | KL learner-state `single_expert` | baseline PPO `SARE` | KL learner-state `SARE` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `original` | `7` | `0.7031` | `1.0000` | `1.0000` | `0.0000` | `1.0000` |
| `original` | `11` | `0.0000` | `0.0000` | `1.0000` | `0.0000` | `0.5625` |
| `original` | `19` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `0.5781` |
| `fresh` | `23` | `0.0000` | `0.0000` | `0.4062` | `0.0000` | `1.0000` |
| `fresh` | `29` | `0.0000` | `0.6250` | `1.0000` | `0.0000` | `1.0000` |
| `fresh` | `31` | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `1.0000` |
| `fresh_extra` | `37` | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `1.0000` |
| `fresh_extra` | `41` | `0.0000` | `0.0000` | `0.4375` | `0.0000` | `1.0000` |
| `fresh_extra` | `43` | `0.0000` | `0.0000` | `1.0000` | `0.0000` | `0.4688` |

Mean greedy success on that structured slice:

- `KL` learner-state `token_dense`: `0.5139`
- `KL` learner-state `single_expert`: `0.7604`
- `KL` learner-state `SARE`: `0.8455`

On the final fresh matched DoorKey block:

| Seed | recovered `token_dense` | KL learner-state `token_dense` | KL learner-state `single_expert` | baseline PPO `SARE` | KL learner-state `SARE` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `47` | `1.0000` | `1.0000` | `0.4531` | `0.0000` | `0.0000` |
| `53` | `1.0000` | `1.0000` | `0.5156` | `0.0000` | `0.5156` |
| `59` | `1.0000` | `1.0000` | `0.4219` | `0.0000` | `0.4219` |

Mean greedy success on that block:

- recovered `token_dense`: `1.0000`
- `KL` learner-state `token_dense`: `1.0000`
- `KL` learner-state `single_expert`: `0.4635`
- `KL` learner-state `SARE`: `0.3125`

Across the final combined DoorKey picture:

| Variant | Mean Greedy Success | Complete-Seed Failures |
| --- | ---: | ---: |
| recovered `token_dense` | `0.5586` | `5` |
| `KL` learner-state `token_dense` | `0.6354` | `4` |
| `KL` learner-state `single_expert` | `0.6862` | `1` |
| baseline PPO `SARE` | `0.0000` | `12` |
| `KL` learner-state `SARE` | `0.7122` | `1` |

So the right claim after the forensic-atlas pass is:

- teacher-guided KL learner-state supervision helps structured students generally
- `SARE` still ends slightly ahead of the matched structured controls in the full combined DoorKey picture
- but the weak final block still does not expose one clean resume-worthy mechanism after the deeper forensic package
- the evidence is no longer strong enough to justify a retry or strengthen the claim into a specifically multi-expert routed DoorKey edge

## Route Dependence

The recovered DoorKey `SARE` checkpoints still look routed, and the current phase adds broader causal evidence that performance depends on routing even though the overall claim stays bounded.

Source artifacts:

- [lss_new_case_route_integrity_report.md](outputs/reports/lss_new_case_route_integrity_report.md)
- [lss_causal_route_dependence_report.md](outputs/reports/lss_causal_route_dependence_report.md)
- [lss_seed29_route_randomization_forensics.md](outputs/reports/lss_seed29_route_randomization_forensics.md)
- [lss_broader_route_dependence_report.md](outputs/reports/lss_broader_route_dependence_report.md)

Key result:

- baseline PPO `SARE` on seed `23`: greedy success `0.0000`, route entropy `1.3857`, active compute `0.5000`
- KL learner-state `SARE` on seed `23`: greedy success `1.0000`, route entropy `1.3804`, active compute `0.5000`
- bounded causal probes now cover seeds `7`, `19`, `23`, `29`, `31`, and `37`
  - fixed-router override has mean greedy-success drop `0.9297`
  - worst expert ablation has mean greedy-success drop `0.9297`
  - current route-randomization has mean greedy-success drop `0.7031`
  - seed `29` is the main narrow exception: repeated current and stronger randomization ladders stay high there, even though fixed-pair overrides still collapse success

So the DoorKey result is not just statistically non-collapsed routing. Under the bounded probe family used here, recovered performance remains causally routing-dependent, but the mechanism claim is narrower than a clean “all route randomization kills performance” story.

## Transfer Check

The exact same method still does not transfer under the bounded KeyCorridor check.

Source artifact: [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md)

All three KeyCorridor seeds stayed flat:

- recovered `token_dense`: `0.0000`, `0.0000`, `0.0000`
- baseline PPO `SARE`: `0.0000`, `0.0000`, `0.0000`
- KL learner-state `SARE`: `0.0000`, `0.0000`, `0.0000`

## Recommendation

- Allow thaw consideration within DoorKey only for the gate-cleared `post_unlock_weighted` candidate pack.
- Keep the scope explicit:
  - teacher-guided extraction only
  - KL learner-state only
  - DoorKey only
  - external `64`-episode evaluation only
- Keep the disallowed claims explicit:
  - no PPO-only routed win
  - no specifically multi-expert routed advantage
  - no cross-task routed advantage
  - no KeyCorridor transfer claim
- Future work must still use the manifest and automated gate rather than narrative override:
  - `configs/claims/doorkey_frozen_claim.yaml`
  - [frozen_benchmark_pack.json](outputs/reports/frozen_benchmark_pack.json)
  - [long_campaign_candidate_pack.json](outputs/reports/long_campaign_candidate_pack.json)
  - [long_campaign_gate_report.md](outputs/reports/long_campaign_gate_report.md)
