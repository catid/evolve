# Pack-Based Claim Gate Dry Run

- frozen benchmark pack: `outputs/reports/round6_current_benchmark_pack.json`
- candidate result pack: `outputs/reports/portfolio_candidate_pack.json`

## Frozen Pack Validation

| Check | Result | Detail |
| --- | --- | --- |
| pack_type | `PASS` | pack type is `frozen_benchmark_pack` |
| provenance.sealed_source_commit | `PASS` | sealed source commit `251c92bc33d7b9ec5aa313dcbdd1a9c245bfa7fe` is well formed |
| provenance.sealed_source_dirty | `PASS` | sealed source dirty flag is `True` |
| manifest_hash | `PASS` | manifest hash matches `e9a141dd6d655b122a63b3504484ae4d67af9519405da134c45e3bfb3302d2c6` |
| schema_version | `PASS` | pack schema version `1` is recognized |
| claim | `PASS` | sealed `claim` matches the manifest |
| canonical_method | `PASS` | sealed `canonical_method` matches the manifest |
| evaluation | `PASS` | sealed `evaluation` matches the manifest |
| seed_groups | `PASS` | sealed `seed_groups` matches the manifest |
| variants | `PASS` | sealed `variants` matches the manifest |
| thresholds | `PASS` | sealed `thresholds` matches the manifest |
| thaw_gate | `PASS` | sealed `thaw_gate` matches the manifest |
| authoritative_artifacts | `PASS` | all required artifact keys are present |
| artifact_hash::baseline_sync | `PASS` | artifact `outputs/reports/next_round_baseline_sync.md` hash matches `8ebcf0635bcb86185bc00c6daeca5183642d6a9e7e7c2963cbb1ebcf72371bd7` |
| artifact_hash::claim_ledger_snapshot | `PASS` | artifact `outputs/reports/claim_ledger_round6_current.md` hash matches `ee26c73a78b1ca070a13df98bc508e3e6480dc17b9bae3ce0ab026ce76a8100d` |
| artifact_hash::next_mega_antiregression | `PASS` | artifact `outputs/reports/next_mega_portfolio_stage5_antiregression.md` hash matches `46fa6a07dcef3bc20383b03e45357c71c2ba532491d78e013f2a737a529f1982` |
| artifact_hash::next_mega_decision_memo | `PASS` | artifact `outputs/reports/next_mega_portfolio_decision_memo.md` hash matches `4a9c3d76e31a6906531ca0a75963a64383dfd9b8eb1b9eab23228e9409731597` |
| artifact_hash::next_mega_gate_report | `PASS` | artifact `outputs/reports/next_mega_portfolio_gate_report.md` hash matches `d107e363f89d91be59ba5ad378e51b42d1a337badfdc9b0a103549fe26db862d` |
| artifact_hash::next_mega_holdout | `PASS` | artifact `outputs/reports/next_mega_portfolio_stage4_holdout.md` hash matches `a30ef1d2a2ea9d6ad445c9548c52f9d640b9f355b796a87a8faefe8405e42f64` |
| artifact_hash::next_mega_route_validation | `PASS` | artifact `outputs/reports/next_mega_portfolio_stage6_route_validation.md` hash matches `c662b19e17d1c3effc23058b77efb0c4765878ee6a5648e03b96ad7455371d92` |
| artifact_hash::next_mega_stability | `PASS` | artifact `outputs/reports/next_mega_portfolio_stage7_stability.md` hash matches `615d6461dfc978a4d8d8a3e438f59079e30276f7a5044cc7dd7d75ab852efd2f` |
| artifact_hash::next_mega_stage1_exploratory | `PASS` | artifact `outputs/reports/next_mega_portfolio_stage1_screening_exploratory.md` hash matches `970654a48f3d5aa20ad7c1d2cad713a0e5d7760cd769b32060fcf2a1963efecc` |
| artifact_hash::next_mega_stage1_fruitful | `PASS` | artifact `outputs/reports/next_mega_portfolio_stage1_screening_fruitful.md` hash matches `64e7cbe3ac8283bb8ecb88644eaac0d5bb48e763c264096af6057aeffba03914` |
| artifact_hash::state_reconciliation | `PASS` | artifact `outputs/reports/next_round_state_reconciliation.md` hash matches `bb5148f3b98a24f5784f3d426b7fb2c3bbb3a81f4c1bcc0bcbe546a51fb6c388` |

## Candidate Pack Validation

| Check | Result | Detail |
| --- | --- | --- |
| pack_type | `PASS` | pack type is `candidate_result_pack` |
| schema_version | `PASS` | candidate schema version `1` is recognized |
| candidate_name | `PASS` | candidate name is `round6` |
| frozen_pack_reference.path | `PASS` | candidate references `outputs/reports/round6_current_benchmark_pack.json` |
| frozen_pack_reference.sha256 | `PASS` | candidate references frozen-pack hash `bddb622d60ac14600327acfe56480a3f0e22e73acd24da049cd278e98d9514c7` |
| frozen_pack_reference.claim_id | `PASS` | candidate claim id matches `doorkey_round6_current_claim` |
| task | `PASS` | candidate task matches `DoorKey` |
| evaluation | `PASS` | candidate uses the required DoorKey external-64 evaluation path |
| requested_claims | `PASS` | requested claim keys are well formed |
| controls_present.type | `PASS` | controls_present is a list of strings |
| controls_present | `PASS` | candidate includes all required fairness controls |
| metrics.combined.type | `PASS` | combined metrics are a mapping |
| metrics.retry_block.type | `PASS` | retry-block metrics are a mapping |
| metrics.combined | `PASS` | combined metrics expose every required variant field |
| metrics.retry_block | `PASS` | retry-block metrics expose every required variant field |
| actual_sets.combined | `PASS` | candidate combined lane/seed set matches the frozen benchmark |
| actual_sets.retry_block | `PASS` | candidate retry-block lane/seed set matches the frozen benchmark |
| artifacts.duplicates | `PASS` | artifact roles are unique |
| artifacts | `PASS` | candidate pack exposes all required artifact roles |
| artifact_hash::candidate_metrics_json | `PASS` | candidate artifact `outputs/reports/hard_family_saturation_candidate_metrics.json` hash matches `3ae5f73f05fc570d149c0ab6545703283510c6ecdeb7507fd3b02356861856b8` |
| artifact_hash::candidate_summary_markdown | `PASS` | candidate artifact `outputs/reports/hard_family_saturation_candidate_summary.md` hash matches `98eede059cabe12591c4bb9b61918fe7c339f378da158d6397b322ecfc95e405` |
| artifact_hash::combined_report_csv | `PASS` | candidate artifact `outputs/reports/hard_family_saturation_candidate_combined_report.csv` hash matches `1e676f86903ad8b10901d01c1963bb2572966dc425c057e7194af639ff1e1481` |
| artifact_hash::combined_report_markdown | `PASS` | candidate artifact `outputs/reports/hard_family_saturation_candidate_combined_report.md` hash matches `d4ee90580be5c466626674124862c8d5b0d858c11450dafdb3def68c649d2cee` |
| artifact_hash::retry_block_report_csv | `PASS` | candidate artifact `outputs/reports/hard_family_saturation_candidate_retry_block_report.csv` hash matches `e63732861aeb3158e6ecec7824457f79aefa97015ab798a3da49f779399515ed` |
| artifact_hash::retry_block_report_markdown | `PASS` | candidate artifact `outputs/reports/hard_family_saturation_candidate_retry_block_report.md` hash matches `a758621f40f4b4700fd506b33da7ae28caf5916851ddb15c72d102a72d8e2867` |
| artifact_consistency::candidate_metrics_json | `PASS` | candidate pack matches the candidate_metrics_json artifact on task, evaluation, controls, metrics, and actual sets |
| provenance.git_commit | `PASS` | candidate provenance git_commit `251c92bc33d7b9ec5aa313dcbdd1a9c245bfa7fe` is well formed |
| provenance.git_dirty | `PASS` | candidate provenance git_dirty is `True` |

## Claim Gate Checks

| Check | Result | Detail |
| --- | --- | --- |
| evaluation_shape | `PASS` | candidate evaluation is a mapping |
| task | `PASS` | candidate task matches `DoorKey` |
| evaluation_path | `PASS` | candidate uses `external_policy_diagnostics` with `64` episodes |
| fairness_controls_shape | `PASS` | controls_present is a list of strings |
| fairness_controls | `PASS` | all required structured controls are present |
| claim_scope_shape | `PASS` | requested_claims is a list of strings |
| claim_scope | `PASS` | candidate stays inside the frozen claim envelope |
| candidate_metrics_shape | `PASS` | metrics is a mapping |
| candidate_metrics_combined_shape | `PASS` | metrics.combined is a mapping |
| candidate_metrics_retry_block_shape | `PASS` | metrics.retry_block is a mapping |
| candidate_metrics | `PASS` | candidate exposes retry-block and combined metrics for required variants |
| retry_block_improvement | `PASS` | candidate retry-block SARE mean `1.0000` exceeds frozen baseline `0.3125` |
| retry_block_vs_single_expert | `PASS` | candidate retry-block SARE mean `1.0000` matches or beats same-block single_expert `1.0000` |
| retry_block_failures | `PASS` | candidate retry-block SARE complete-seed failures `0` stay within gate `1` |
| combined_picture_mean | `PASS` | candidate combined SARE mean `1.0000` preserves or improves frozen baseline `0.7122` |
| combined_picture_failures | `PASS` | candidate combined SARE complete-seed failures `0` stay within gate `1` |

## Verdict

PASS: thaw consideration allowed
