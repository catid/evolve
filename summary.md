# Current Summary

## Current Claim State

- The frozen benchmark pack remains the baseline DoorKey comparison unit.
- The staged long-horizon candidate `post_unlock_weighted` now clears the existing pack-based gate, so thaw consideration is allowed within DoorKey only.
- The post-pass qualification campaign does not upgrade that candidate into the canonical DoorKey benchmark; it remains thaw-qualified but not canonical.
- The hard-block canonization campaign strengthens that same outcome: bounded hard-block fixes improved the difficult family, but none made KL learner-state `SARE` stop trailing matched KL learner-state `token_dense` on the `post_pass_b` plus `post_pass_c` family.
- The longer hard-family dev/holdout program leaves that same status in place: the best dev candidate still trailed matched KL learner-state `token_dense` on the `post_pass_b` plus `post_pass_c` development split, so the program stopped at Stage 3 before holdout testing and again left the candidate thaw-qualified but not canonical.
- Allowed current scope:
  - teacher-guided KL learner-state DoorKey `SARE` result
  - external `64`-episode evaluation only
- Still not allowed:
  - PPO-only routed win
  - specifically multi-expert routed advantage
  - cross-task routed advantage
  - KeyCorridor transfer claim
- The baseline thaw thresholds are still defined by the frozen pack:
  - retry-block KL learner-state `SARE` mean must beat `0.3125` on `47/53/59`
  - retry-block KL learner-state `SARE` must at least match same-block KL learner-state `single_expert`
  - combined DoorKey KL learner-state `SARE` mean must stay at or above `0.7122`
- Baseline and gate artifacts: [frozen_claim_envelope.md](outputs/reports/frozen_claim_envelope.md), [frozen_benchmark_pack.md](outputs/reports/frozen_benchmark_pack.md), [frozen_benchmark_pack_validation.md](outputs/reports/frozen_benchmark_pack_validation.md), [claim_gate_pack_dry_run.md](outputs/reports/claim_gate_pack_dry_run.md), and [freeze_hardening_operational_memo.md](outputs/reports/freeze_hardening_operational_memo.md).
- Gate-cleared candidate artifacts: [long_campaign_candidate_pack.json](outputs/reports/long_campaign_candidate_pack.json), [long_campaign_gate_report.md](outputs/reports/long_campaign_gate_report.md), and [long_campaign_decision_memo.md](outputs/reports/long_campaign_decision_memo.md).
- Post-pass qualification artifacts: [post_pass_stage1_fresh_blocks.md](outputs/reports/post_pass_stage1_fresh_blocks.md), [post_pass_stage2_full_fairness.md](outputs/reports/post_pass_stage2_full_fairness.md), [post_pass_stage3_route_validation.md](outputs/reports/post_pass_stage3_route_validation.md), [post_pass_stage4_longitudinal_stability.md](outputs/reports/post_pass_stage4_longitudinal_stability.md), [post_pass_successor_pack_draft.md](outputs/reports/post_pass_successor_pack_draft.md), [post_pass_gate_report.md](outputs/reports/post_pass_gate_report.md), and [post_pass_canonization_decision_memo.md](outputs/reports/post_pass_canonization_decision_memo.md).
- Hard-block canonization artifacts: [canonization_campaign_registration.md](outputs/reports/canonization_campaign_registration.md), [canonization_stage2_hard_block_screening.md](outputs/reports/canonization_stage2_hard_block_screening.md), [canonization_stage3_hard_block_fairness.md](outputs/reports/canonization_stage3_hard_block_fairness.md), [canonization_gate_report.md](outputs/reports/canonization_gate_report.md), and [canonization_decision_memo.md](outputs/reports/canonization_decision_memo.md).
- Long hard-family program artifacts: [hard_family_definition.md](outputs/reports/hard_family_definition.md), [hard_family_campaign_registration.md](outputs/reports/hard_family_campaign_registration.md), [hard_family_stage2_dev_screening.md](outputs/reports/hard_family_stage2_dev_screening.md), [hard_family_stage3_fairness.md](outputs/reports/hard_family_stage3_fairness.md), [hard_family_gate_report.md](outputs/reports/hard_family_gate_report.md), and [hard_family_canonization_decision_memo.md](outputs/reports/hard_family_canonization_decision_memo.md).

- `flat_dense` remains the strongest greedy DoorKey control.
- recovered `token_dense` with `ppo.ent_coef=0.001` remains the canonical tokenized control.
- PPO-only `SARE` is still negative on greedy DoorKey.
- The forensic atlas still matters because it constrained the intervention space:
  - seed `47` looked more route-fragile
  - seeds `53/59` looked more like shared post-unlock structured-student collapse
  - that evidence motivated the winning bounded intervention family: post-unlock KL weighting inside the existing learner-state path
  - see [lss_forensic_casebook.md](outputs/reports/lss_forensic_casebook.md), [lss_forensic_round_audit.md](outputs/reports/lss_forensic_round_audit.md), [lss_forensic_route_locality.md](outputs/reports/lss_forensic_route_locality.md), and [lss_resume_qualification_scorecard.md](outputs/reports/lss_resume_qualification_scorecard.md)
- The long-horizon campaign then produced the first gate-cleared within-envelope DoorKey candidate:
  - weak-block Stage 2 kept only `post_unlock_weighted`; the recency and phase-balanced variants all failed at `0.0000` mean on `47/53/59`
  - Stage 3 raised weak-block KL learner-state `SARE` mean to `0.4635`, exactly matching same-block KL learner-state `single_expert`
  - Stage 4 preserved and improved the combined DoorKey picture:
    - `KL` learner-state `token_dense`: `0.6354`
    - `KL` learner-state `single_expert`: `0.6862`
    - `KL` learner-state `SARE`: `0.7500`
  - Stage 5 kept meaningful route dependence on the selected weak and strong cases
  - Stage 6 packaged the candidate and the gate returned `PASS: thaw consideration allowed`
  - see [long_campaign_stage2_screening.md](outputs/reports/long_campaign_stage2_screening.md), [long_campaign_stage3_fairness.md](outputs/reports/long_campaign_stage3_fairness.md), [long_campaign_stage4_replication.md](outputs/reports/long_campaign_stage4_replication.md), [long_campaign_stage5_route_validation.md](outputs/reports/long_campaign_stage5_route_validation.md), [long_campaign_gate_report.md](outputs/reports/long_campaign_gate_report.md), and [long_campaign_decision_memo.md](outputs/reports/long_campaign_decision_memo.md)
- The post-pass qualification campaign kept the candidate alive but did not canonize it:
  - fresh-block expansion stayed clearly above matched `single_expert`, but `post_pass_b` remained token-dense-led
  - expanded fairness, route validation, and longitudinal stability all passed
  - the successor draft is coherent and the gate still returns `PASS: thaw consideration allowed`
  - the final status is `remains thaw-qualified but not canonical`
  - see [post_pass_stage1_fresh_blocks.md](outputs/reports/post_pass_stage1_fresh_blocks.md), [post_pass_stage2_full_fairness.md](outputs/reports/post_pass_stage2_full_fairness.md), [post_pass_stage3_route_validation.md](outputs/reports/post_pass_stage3_route_validation.md), [post_pass_stage4_longitudinal_stability.md](outputs/reports/post_pass_stage4_longitudinal_stability.md), [post_pass_successor_pack_draft.md](outputs/reports/post_pass_successor_pack_draft.md), [post_pass_gate_report.md](outputs/reports/post_pass_gate_report.md), and [post_pass_canonization_decision_memo.md](outputs/reports/post_pass_canonization_decision_memo.md)
- The hard-block canonization campaign did not move that status:
  - `post_unlock_weighted_round5` and `post_unlock_weighted_disagreement075` both improved the hard-block family relative to the current candidate
  - but neither stopped KL learner-state `SARE` from trailing matched KL learner-state `token_dense` on the hard-block family
  - the campaign therefore stopped at Stage 3, did not produce a successor pack, and left the candidate `thaw-qualified but not canonical`
  - see [canonization_stage2_hard_block_screening.md](outputs/reports/canonization_stage2_hard_block_screening.md), [canonization_stage3_hard_block_fairness.md](outputs/reports/canonization_stage3_hard_block_fairness.md), [canonization_gate_report.md](outputs/reports/canonization_gate_report.md), and [canonization_decision_memo.md](outputs/reports/canonization_decision_memo.md)
- The long hard-family dev/holdout program also did not move that status:
  - `fresh_final` was promoted into an explicit withheld hard-family test split while `post_pass_b/post_pass_c` remained the development family
  - the new phase-balanced round-5 descendants collapsed on the dev split
  - the surviving `round5` and `disagreement075` lines still failed the dev fairness bar against matched `token_dense`
  - the program therefore stopped at Stage 3, produced explicit stop-path holdout/anti-regression/route/stability artifacts, and again left the candidate `thaw-qualified but not canonical`
  - see [hard_family_definition.md](outputs/reports/hard_family_definition.md), [hard_family_stage2_dev_screening.md](outputs/reports/hard_family_stage2_dev_screening.md), [hard_family_stage3_fairness.md](outputs/reports/hard_family_stage3_fairness.md), [hard_family_gate_report.md](outputs/reports/hard_family_gate_report.md), and [hard_family_canonization_decision_memo.md](outputs/reports/hard_family_canonization_decision_memo.md)
- The recovered DoorKey `SARE` policy remains causally routing-dependent under bounded eval-time probes:
  - expert ablation and fixed-router override remain strongly harmful across the expanded recovered-seed set
  - route randomization is catastrophic on most recovered seeds, but seed `29` is now a genuine narrow exception rather than a weak-probe artifact
  - see [lss_seed29_route_randomization_forensics.md](outputs/reports/lss_seed29_route_randomization_forensics.md) and [lss_broader_route_dependence_report.md](outputs/reports/lss_broader_route_dependence_report.md)
- The exact same method shows no bounded KeyCorridor transfer. See [lss_keycorridor_transfer_report.md](outputs/reports/lss_keycorridor_transfer_report.md).
- The current repo recommendation is to allow thaw consideration within DoorKey only for the gate-cleared long-campaign candidate, while still requiring the automated gate for any future candidate. See [long_campaign_decision_memo.md](outputs/reports/long_campaign_decision_memo.md) and [freeze_hardening_operational_memo.md](outputs/reports/freeze_hardening_operational_memo.md).

## Claim Gate

- Validate the sealed frozen benchmark pack first: [frozen_benchmark_pack_validation.md](outputs/reports/frozen_benchmark_pack_validation.md)
- Package any future thaw candidate using: [candidate_result_pack_schema.md](outputs/reports/candidate_result_pack_schema.md) and [candidate_result_pack_template.json](outputs/reports/candidate_result_pack_template.json)
- The current frozen reference candidate still fails the pack-based gate: [claim_gate_pack_dry_run.md](outputs/reports/claim_gate_pack_dry_run.md)
- The current long-campaign candidate now passes the pack-based gate: [long_campaign_gate_report.md](outputs/reports/long_campaign_gate_report.md)
- Malformed or incomplete candidates should be treated as missing prerequisites: [claim_gate_pack_inconclusive.md](outputs/reports/claim_gate_pack_inconclusive.md)
- The adversarial corpus now exercises PASS / FAIL / INCONCLUSIVE paths and is the conformance reference for future gate changes: [claim_gate_corpus_report.md](outputs/reports/claim_gate_corpus_report.md) and [claim_gate_conformance_report.md](outputs/reports/claim_gate_conformance_report.md)
- Ad hoc markdown reports are insufficient for thaw consideration; the candidate pack plus pack-based gate are the only thaw triage surface. See [claim_gate_redteam_decision_memo.md](outputs/reports/claim_gate_redteam_decision_memo.md).
- The gate is now also replay-checked against the repo's real historical claim phases, not just synthetic red-team cases. See [historical_candidate_pack_catalog.md](outputs/reports/historical_candidate_pack_catalog.md), [claim_history_replay_report.md](outputs/reports/claim_history_replay_report.md), [claim_ledger_consistency_audit.md](outputs/reports/claim_ledger_consistency_audit.md), and [claim_history_replay_decision_memo.md](outputs/reports/claim_history_replay_decision_memo.md).
