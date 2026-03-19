# DoorKey Extraction Report

## Current Conclusion

- `flat_dense` remains the strongest greedy DoorKey control.
- recovered `token_dense` (`ppo.ent_coef=0.001`) remains the canonical tokenized control.
- PPO-only `SARE` still loses the fair greedy DoorKey comparison.
- The frozen benchmark pack remains the baseline comparison unit for DoorKey thaw work.
- The current active DoorKey benchmark now sits inside the same teacher-guided `KL` learner-state family:
  - benchmark: `round6`
  - lineage: successor to `post_unlock_weighted`
  - scope: DoorKey only
  - evaluation: external `64`-episode `policy_diagnostics`
  - gate result: `PASS: thaw consideration allowed`
  - current benchmark result: `round6 confirmed as active DoorKey benchmark and internal DoorKey envelope strengthened`
  - current portfolio confirmation: `round6 confirmed as active DoorKey benchmark and internal DoorKey benchmark state strengthened`
  - exploratory adjacent-task result: `KeyCorridor clearly negative`
  - legacy baseline: the frozen benchmark pack remains archived for provenance and historical comparison
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
- post-pass successor draft: [post_pass_successor_pack_draft.md](outputs/reports/post_pass_successor_pack_draft.md)
- post-pass candidate pack gate result: [post_pass_gate_report.md](outputs/reports/post_pass_gate_report.md)
- post-pass canonization decision: [post_pass_canonization_decision_memo.md](outputs/reports/post_pass_canonization_decision_memo.md)
- hard-block canonization screening: [canonization_stage2_hard_block_screening.md](outputs/reports/canonization_stage2_hard_block_screening.md)
- hard-block canonization fairness: [canonization_stage3_hard_block_fairness.md](outputs/reports/canonization_stage3_hard_block_fairness.md)
- hard-block canonization gate status: [canonization_gate_report.md](outputs/reports/canonization_gate_report.md)
- hard-block canonization decision: [canonization_decision_memo.md](outputs/reports/canonization_decision_memo.md)
- hard-family definition: [hard_family_definition.md](outputs/reports/hard_family_definition.md)
- hard-family dev screening: [hard_family_stage2_dev_screening.md](outputs/reports/hard_family_stage2_dev_screening.md)
- hard-family dev fairness: [hard_family_stage3_fairness.md](outputs/reports/hard_family_stage3_fairness.md)
- hard-family decision: [hard_family_canonization_decision_memo.md](outputs/reports/hard_family_canonization_decision_memo.md)
- saturation definition: [hard_family_saturation_definition.md](outputs/reports/hard_family_saturation_definition.md)
- saturation screening: [hard_family_saturation_stage1_screening.md](outputs/reports/hard_family_saturation_stage1_screening.md)
- saturation fairness: [hard_family_saturation_stage2_fairness.md](outputs/reports/hard_family_saturation_stage2_fairness.md)
- saturation holdout: [hard_family_saturation_stage3_holdout.md](outputs/reports/hard_family_saturation_stage3_holdout.md)
- saturation anti-regression: [hard_family_saturation_stage4_antiregression.md](outputs/reports/hard_family_saturation_stage4_antiregression.md)
- saturation route validation: [hard_family_saturation_stage5_route_validation.md](outputs/reports/hard_family_saturation_stage5_route_validation.md)
- saturation stability: [hard_family_saturation_stage6_stability.md](outputs/reports/hard_family_saturation_stage6_stability.md)
- saturation successor pack: [hard_family_saturation_successor_pack.md](outputs/reports/hard_family_saturation_successor_pack.md)
- saturation gate result: [hard_family_saturation_gate_report.md](outputs/reports/hard_family_saturation_gate_report.md)
- saturation decision: [hard_family_saturation_decision_memo.md](outputs/reports/hard_family_saturation_decision_memo.md)
- migration registration: [successor_migration_registration.md](outputs/reports/successor_migration_registration.md)
- migration screening: [successor_migration_stage1_screening.md](outputs/reports/successor_migration_stage1_screening.md)
- migration holdout: [successor_migration_stage3_holdout.md](outputs/reports/successor_migration_stage3_holdout.md)
- migration route validation: [successor_migration_stage5_route_validation.md](outputs/reports/successor_migration_stage5_route_validation.md)
- migration stability: [successor_migration_stage6_stability.md](outputs/reports/successor_migration_stage6_stability.md)
- migration candidate pack: [successor_migration_candidate_pack.json](outputs/reports/successor_migration_candidate_pack.json)
- migration gate result: [successor_migration_gate_report.md](outputs/reports/successor_migration_gate_report.md)
- migration decision: [successor_migration_decision_memo.md](outputs/reports/successor_migration_decision_memo.md)
- mega-league registration: [successor_mega_league_registration.md](outputs/reports/successor_mega_league_registration.md)
- mega-league screening: [successor_mega_league_stage1_screening.md](outputs/reports/successor_mega_league_stage1_screening.md)
- mega-league verification: [successor_mega_league_stage2_verification.md](outputs/reports/successor_mega_league_stage2_verification.md)
- expansion program registration: [expansion_mega_program_registration.md](outputs/reports/expansion_mega_program_registration.md)
- expansion program screening: [expansion_mega_program_stage1_screening.md](outputs/reports/expansion_mega_program_stage1_screening.md)
- expansion program verification: [expansion_mega_program_stage2_verification.md](outputs/reports/expansion_mega_program_stage2_verification.md)
- expansion program fairness: [expansion_mega_program_stage3_fairness.md](outputs/reports/expansion_mega_program_stage3_fairness.md)
- expansion program holdout: [expansion_mega_program_stage4_holdout.md](outputs/reports/expansion_mega_program_stage4_holdout.md)
- expansion program anti-regression: [expansion_mega_program_stage5_antiregression.md](outputs/reports/expansion_mega_program_stage5_antiregression.md)
- expansion program route validation: [expansion_mega_program_stage6_route_validation.md](outputs/reports/expansion_mega_program_stage6_route_validation.md)
- expansion program stability: [expansion_mega_program_stage7_stability.md](outputs/reports/expansion_mega_program_stage7_stability.md)
- active benchmark pack: [portfolio_candidate_pack.json](outputs/reports/portfolio_candidate_pack.json)
- current gate result: [portfolio_gate_report.md](outputs/reports/portfolio_gate_report.md)
- current decision: [portfolio_decision_memo.md](outputs/reports/portfolio_decision_memo.md)
- mega-league gate result: [successor_mega_league_gate_report.md](outputs/reports/successor_mega_league_gate_report.md)
- mega-league decision: [successor_mega_league_decision_memo.md](outputs/reports/successor_mega_league_decision_memo.md)
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
- The post-pass qualification campaign kept that gate `PASS`, but did not clear the canonization bar because the candidate still trails matched `token_dense` on the harder new fresh block `post_pass_b`. See [post_pass_stage1_fresh_blocks.md](outputs/reports/post_pass_stage1_fresh_blocks.md), [post_pass_stage2_full_fairness.md](outputs/reports/post_pass_stage2_full_fairness.md), [post_pass_stage3_route_validation.md](outputs/reports/post_pass_stage3_route_validation.md), [post_pass_stage4_longitudinal_stability.md](outputs/reports/post_pass_stage4_longitudinal_stability.md), [post_pass_successor_pack_draft.md](outputs/reports/post_pass_successor_pack_draft.md), [post_pass_gate_report.md](outputs/reports/post_pass_gate_report.md), and [post_pass_canonization_decision_memo.md](outputs/reports/post_pass_canonization_decision_memo.md).
- The hard-block canonization campaign then attacked that exact blocker with three bounded hard-block interventions. `post_unlock_weighted_round5` improved the hard-block family the most, but even it still trailed matched KL learner-state `token_dense` on `post_pass_b` and on the full `post_pass_b` plus `post_pass_c` family, so the campaign stopped at Stage 3 and did not produce a successor candidate pack. See [canonization_stage2_hard_block_screening.md](outputs/reports/canonization_stage2_hard_block_screening.md), [canonization_stage3_hard_block_fairness.md](outputs/reports/canonization_stage3_hard_block_fairness.md), [canonization_gate_report.md](outputs/reports/canonization_gate_report.md), and [canonization_decision_memo.md](outputs/reports/canonization_decision_memo.md).
- The longer hard-family dev/holdout program then promoted `fresh_final` into an explicit withheld hard-family test while keeping `post_pass_b/post_pass_c` as the development family. Two new phase-balanced round-5 descendants collapsed on the dev split, and the two surviving older candidates still failed the split-wide fairness bar:
  - `post_unlock_weighted_round5`: dev KL learner-state `SARE` `0.8464` vs matched `token_dense` `0.9453`
  - `post_unlock_weighted_disagreement075`: dev KL learner-state `SARE` `0.6875` vs matched `token_dense` `0.8333`
  so the program stopped at Stage 3 before holdout testing and again left the accepted state `thaw-qualified but not canonical`. See [hard_family_definition.md](outputs/reports/hard_family_definition.md), [hard_family_stage2_dev_screening.md](outputs/reports/hard_family_stage2_dev_screening.md), [hard_family_stage3_fairness.md](outputs/reports/hard_family_stage3_fairness.md), and [hard_family_canonization_decision_memo.md](outputs/reports/hard_family_canonization_decision_memo.md).
- The saturation-scale hard-family program turned that boundary into a canonization-qualified result. It expanded the hard family to three development blocks and two holdout blocks, screened twenty bounded candidates across ten mechanism directions, and advanced only `round6` through the full verification lane:
  - hard-family dev KL learner-state `SARE`: `1.0000` vs matched `token_dense` `1.0000`
  - hard-family holdout KL learner-state `SARE`: `1.0000` vs matched `token_dense` `1.0000`
  - healthy-block KL learner-state `SARE`: `1.0000` vs matched `token_dense` `0.9141`
  - route validation: `pass`
  - stability: `stable_plateau` on the selected dev, holdout, and healthy probes
  - successor pack gate verdict: `PASS: thaw consideration allowed`
  so the accepted status became `qualified for canonization within DoorKey`. See [hard_family_saturation_definition.md](outputs/reports/hard_family_saturation_definition.md), [hard_family_saturation_stage1_screening.md](outputs/reports/hard_family_saturation_stage1_screening.md), [hard_family_saturation_stage2_fairness.md](outputs/reports/hard_family_saturation_stage2_fairness.md), [hard_family_saturation_stage3_holdout.md](outputs/reports/hard_family_saturation_stage3_holdout.md), [hard_family_saturation_stage4_antiregression.md](outputs/reports/hard_family_saturation_stage4_antiregression.md), [hard_family_saturation_stage5_route_validation.md](outputs/reports/hard_family_saturation_stage5_route_validation.md), [hard_family_saturation_stage6_stability.md](outputs/reports/hard_family_saturation_stage6_stability.md), [hard_family_saturation_gate_report.md](outputs/reports/hard_family_saturation_gate_report.md), and [hard_family_saturation_decision_memo.md](outputs/reports/hard_family_saturation_decision_memo.md).

- The post-canonization migration and challenger league then answered the remaining operational question. It ran twelve bounded challengers across six mechanism directions on two development families, and none displaced `round6` cleanly enough to reach matched fairness. The program then shifted to migration readiness, where `round6` held the holdout and healthy-block pictures, passed the corrected dev/holdout/healthy route probes, remained `stable_plateau`, and preserved the gate `PASS`, so `round6` became the active DoorKey benchmark while the frozen pack remained archived as the legacy baseline. See [successor_migration_registration.md](outputs/reports/successor_migration_registration.md), [successor_migration_stage1_screening.md](outputs/reports/successor_migration_stage1_screening.md), [successor_migration_stage3_holdout.md](outputs/reports/successor_migration_stage3_holdout.md), [successor_migration_stage4_antiregression.md](outputs/reports/successor_migration_stage4_antiregression.md), [successor_migration_stage5_route_validation.md](outputs/reports/successor_migration_stage5_route_validation.md), [successor_migration_stage6_stability.md](outputs/reports/successor_migration_stage6_stability.md), [successor_migration_gate_report.md](outputs/reports/successor_migration_gate_report.md), and [successor_migration_decision_memo.md](outputs/reports/successor_migration_decision_memo.md).
- The benchmark-expansion mega program then stress-tested that active state instead of widening the claim envelope. It ran thirty challenger runs across fifteen bounded mechanism directions on four DoorKey development families, reran the four tied late-round challengers twice, and still found no fairness survivor against matched controls. The incumbent `round6` then carried the holdout families at `0.9154` versus `0.9167` for both matched `token_dense` and `single_expert`, preserved a `1.0000` healthy-family mean, stayed causally routed on dev/holdout/healthy probes, and remained `stable_plateau` even though one holdout probe was a flat-zero plateau rather than a recovery curve. The fenced KeyCorridor track was strictly negative at `0.0000` for both same-family `SARE` and `token_dense`, so the internal DoorKey benchmark role strengthened while the public envelope stayed narrow. See [expansion_mega_program_state_reconciliation.md](outputs/reports/expansion_mega_program_state_reconciliation.md), [expansion_mega_program_stage1_screening.md](outputs/reports/expansion_mega_program_stage1_screening.md), [expansion_mega_program_stage2_verification.md](outputs/reports/expansion_mega_program_stage2_verification.md), [expansion_mega_program_stage3_fairness.md](outputs/reports/expansion_mega_program_stage3_fairness.md), [expansion_mega_program_stage4_holdout.md](outputs/reports/expansion_mega_program_stage4_holdout.md), [expansion_mega_program_stage5_antiregression.md](outputs/reports/expansion_mega_program_stage5_antiregression.md), [expansion_mega_program_stage6_route_validation.md](outputs/reports/expansion_mega_program_stage6_route_validation.md), [expansion_mega_program_stage7_stability.md](outputs/reports/expansion_mega_program_stage7_stability.md), [expansion_mega_program_stage8_exploratory_transfer.md](outputs/reports/expansion_mega_program_stage8_exploratory_transfer.md), [expansion_mega_program_gate_report.md](outputs/reports/expansion_mega_program_gate_report.md), and [expansion_mega_program_decision_memo.md](outputs/reports/expansion_mega_program_decision_memo.md).
- The later 50/50 portfolio campaign then re-ran the active benchmark question under a broader fruitful/exploratory split rather than a narrower challenger bracket. It spent forty substantive challenger runs split `20/20` fruitful versus exploratory, reran every Stage 1 survivor twice, and still found no matched-fairness survivor. `round6` then held the holdout families at `0.8320`, preserved a `1.0000` healthy-family mean, stayed causally routed on dev/holdout/healthy probes, remained `stable_plateau`, and again stopped short of any KeyCorridor widening because the fenced adjacent-task track was clearly negative. See [portfolio_stage1_screening_fruitful.md](outputs/reports/portfolio_stage1_screening_fruitful.md), [portfolio_stage1_screening_exploratory.md](outputs/reports/portfolio_stage1_screening_exploratory.md), [portfolio_stage2_verification.md](outputs/reports/portfolio_stage2_verification.md), [portfolio_stage3_fairness.md](outputs/reports/portfolio_stage3_fairness.md), [portfolio_stage4_holdout.md](outputs/reports/portfolio_stage4_holdout.md), [portfolio_stage5_antiregression.md](outputs/reports/portfolio_stage5_antiregression.md), [portfolio_stage6_route_validation.md](outputs/reports/portfolio_stage6_route_validation.md), [portfolio_stage7_stability.md](outputs/reports/portfolio_stage7_stability.md), [portfolio_stage8_exploratory_transfer.md](outputs/reports/portfolio_stage8_exploratory_transfer.md), [portfolio_gate_report.md](outputs/reports/portfolio_gate_report.md), and [portfolio_decision_memo.md](outputs/reports/portfolio_decision_memo.md).

The right current wording is now:

- `round6` is the active canonical benchmark within DoorKey only
- `post_unlock_weighted` remains the earlier thaw-qualified predecessor that opened the lane, not the final canonical successor itself
- the hard-block canonization campaign and the earlier hard-family dev/holdout program remain negative waypoints inside that longer lineage, not the final status
- the frozen benchmark pack remains archived as the legacy baseline and provenance anchor, not the active benchmark
- the frozen operational frontier now starts from `round7`, keeps `round10` as the replay-validated alternate, and is guarded by [portfolio_frontier_contract.md](outputs/reports/portfolio_frontier_contract.md), [portfolio_frontier_schedule.md](outputs/reports/portfolio_frontier_schedule.md), [portfolio_frontier_kit.md](outputs/reports/portfolio_frontier_kit.md), [portfolio_seed_pack.md](outputs/reports/portfolio_seed_pack.md), [portfolio_frontier_doctor.md](outputs/reports/portfolio_frontier_doctor.md), [portfolio_seed_pack_doctor.md](outputs/reports/portfolio_seed_pack_doctor.md), and [portfolio_frontier_guard_report.md](outputs/reports/portfolio_frontier_guard_report.md)
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

- Qualify the `round6` successor for canonization within DoorKey only.
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
  - [hard_family_saturation_successor_pack.json](outputs/reports/hard_family_saturation_successor_pack.json)
  - [hard_family_saturation_gate_report.md](outputs/reports/hard_family_saturation_gate_report.md)
