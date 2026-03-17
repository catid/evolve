# Pack-Based Claim Gate Dry Run

- frozen benchmark pack: `outputs/reports/frozen_benchmark_pack.json`
- candidate result pack: `outputs/reports/historical_candidate_packs/multi_expert_hardening/multi_expert_hardening.json`

## Frozen Pack Validation

| Check | Result | Detail |
| --- | --- | --- |
| pack_type | `PASS` | pack type is `frozen_benchmark_pack` |
| provenance.sealed_source_commit | `PASS` | sealed source commit `8d691d69559de8e2aba18481bebcb7151ac15d84` is well formed |
| provenance.sealed_source_dirty | `PASS` | sealed source dirty flag is `True` |
| manifest_hash | `PASS` | manifest hash matches `d10db1cff6f8cb55f434d3d11f0079d8f7e72ea45d2c46adc1c03909dece0bd6` |
| schema_version | `PASS` | pack schema version `1` is recognized |
| claim | `PASS` | sealed `claim` matches the manifest |
| canonical_method | `PASS` | sealed `canonical_method` matches the manifest |
| evaluation | `PASS` | sealed `evaluation` matches the manifest |
| seed_groups | `PASS` | sealed `seed_groups` matches the manifest |
| variants | `PASS` | sealed `variants` matches the manifest |
| thresholds | `PASS` | sealed `thresholds` matches the manifest |
| thaw_gate | `PASS` | sealed `thaw_gate` matches the manifest |
| authoritative_artifacts | `PASS` | all required artifact keys are present |
| artifact_hash::claim_gate_dry_run | `PASS` | artifact `outputs/reports/claim_gate_dry_run.md` hash matches `c38ee9a1df0197f405ed6aa16ad90a27f4d6b8f3d9f960588c0e3bbd70cf8dd7` |
| artifact_hash::claim_gate_dry_run_json | `PASS` | artifact `outputs/reports/claim_gate_dry_run.json` hash matches `152056ccf996def48d342cfb6f98cdea2e30eae915a3333fdcc1721d8db7f41d` |
| artifact_hash::claim_ledger | `PASS` | artifact `outputs/reports/claim_ledger.md` hash matches `8127d6aeb0b06bd69d13dbb75b37f991abebe4ba501fc60cf4a8d0c6a99d691d` |
| artifact_hash::combined_doorkey_csv | `PASS` | artifact `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.csv` hash matches `f0c83d833ef9bfe3fa9e2db97f10e5bc42f6c1ab42805c14b21aa53ca0744b3b` |
| artifact_hash::combined_doorkey_report | `PASS` | artifact `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.md` hash matches `e0cdd2b799538aad926156a19bdf88da19cb8c62a091ab16799a3e94cc87e19d` |
| artifact_hash::final_block_csv | `PASS` | artifact `outputs/reports/lss_final_block_single_expert_control_report.csv` hash matches `8a7c5491529286337be0a727d2a71649ca4804b0e33efe8eba497354f46f7f2c` |
| artifact_hash::final_block_report | `PASS` | artifact `outputs/reports/lss_final_block_single_expert_control_report.md` hash matches `084c6b96a5dfac5536af36009fe08664a42b21a61b230a3a03aa5e48a772812f` |
| artifact_hash::forensic_casebook | `PASS` | artifact `outputs/reports/lss_forensic_casebook.md` hash matches `13d20ae9bfbf0def1595037882c44085c868a10b62c2a9e245e589d14de5f98e` |
| artifact_hash::forensic_decision_memo | `PASS` | artifact `outputs/reports/lss_forensic_atlas_decision_memo.md` hash matches `646d16c2050c200301c9508dd1b579d39607aabc983913ffedc3a7a01b121998` |
| artifact_hash::forensic_round_audit | `PASS` | artifact `outputs/reports/lss_forensic_round_audit.md` hash matches `cb9cb1a8fee052cca94fb6bfb078e1a01f6e7e286585dbea9337a27fb2c4505b` |
| artifact_hash::forensic_route_locality | `PASS` | artifact `outputs/reports/lss_forensic_route_locality.md` hash matches `30ae43a28d514956eebac92c9af81490e04f8c4e1e64b6240ce44f9f1baf026c` |
| artifact_hash::freeze_hardening_decision_memo | `PASS` | artifact `outputs/reports/freeze_hardening_decision_memo.md` hash matches `da0fdc73426b31c2a042bc71cef4a30307364c7c3333d4a3867b0b0b843bc7a5` |
| artifact_hash::frozen_claim_envelope | `PASS` | artifact `outputs/reports/frozen_claim_envelope.md` hash matches `58cd74802397935f8abc2dbaf1119fa59046c3e15bc352deac098ace03e2a9d8` |
| artifact_hash::frozen_validation_csv | `PASS` | artifact `outputs/reports/frozen_baseline_validation.csv` hash matches `3ec73864f9ab0d2cca0b115f9bee267ffd1e448231a15f70eebd8e3a7c03f162` |
| artifact_hash::frozen_validation_json | `PASS` | artifact `outputs/reports/frozen_baseline_validation.json` hash matches `1888fb7ac3e78dead22811d99b88ee1d6b250c37de8fa0600e4bc7ea12a1918b` |
| artifact_hash::frozen_validation_report | `PASS` | artifact `outputs/reports/frozen_baseline_validation.md` hash matches `9ce8e7f5055663abea1ea7444bde72ecc3789b9605a644ce24375f0149bf1248` |
| artifact_hash::future_retry_template | `PASS` | artifact `outputs/reports/future_retry_template.md` hash matches `9ecaa4ac44e4ced4025bbb90959f3158dd3dbecab8b834b618f0e7270eb6e84e` |
| artifact_hash::keycorridor_transfer_csv | `PASS` | artifact `outputs/reports/lss_keycorridor_transfer_report.csv` hash matches `fbc84ede909faa0a7170390451f09ada21a94e674d8b7ee1643effdb14468538` |
| artifact_hash::keycorridor_transfer_report | `PASS` | artifact `outputs/reports/lss_keycorridor_transfer_report.md` hash matches `ee81af42d6e12ee8167c84a531db219627bdc8622913554c79d418be93ac93f9` |
| artifact_hash::manifest_report | `PASS` | artifact `outputs/reports/frozen_claim_manifest_report.md` hash matches `a47eb42e06e4c8ba02e773c099e7a07917a071d2d0511c7aa325f2cc871947b6` |
| artifact_hash::resume_scorecard | `PASS` | artifact `outputs/reports/lss_resume_qualification_scorecard.md` hash matches `843470eb82a25313b09cc796d3ef5f01f9efa6b02dfaf96aaa61e1bc80265028` |

## Candidate Pack Validation

| Check | Result | Detail |
| --- | --- | --- |
| pack_type | `PASS` | pack type is `candidate_result_pack` |
| schema_version | `PASS` | candidate schema version `1` is recognized |
| candidate_name | `PASS` | candidate name is `multi_expert_hardening` |
| frozen_pack_reference.path | `PASS` | candidate references `outputs/reports/frozen_benchmark_pack.json` |
| frozen_pack_reference.sha256 | `PASS` | candidate references frozen-pack hash `21f69b44b2a1567f6432e0ec2a0f5dce2c4f74ea256196d865fbdf384580a873` |
| frozen_pack_reference.claim_id | `PASS` | candidate claim id matches `doorkey_frozen_claim` |
| task | `PASS` | candidate task matches `DoorKey` |
| evaluation | `PASS` | candidate uses the required DoorKey external-64 evaluation path |
| requested_claims | `PASS` | requested claim keys are well formed |
| controls_present.type | `PASS` | controls_present is a list of strings |
| controls_present | `PASS` | candidate includes all required fairness controls |
| metrics.combined.type | `PASS` | combined metrics are a mapping |
| metrics.retry_block.type | `PASS` | retry-block metrics are a mapping |
| metrics.combined | `PASS` | combined metrics expose every required variant field |
| metrics.retry_block | `FAIL` | retry-block metrics are missing fields: `kl_lss_single_expert:<variant>` |
| actual_sets.combined | `PASS` | candidate combined lane/seed set matches the frozen benchmark |
| actual_sets.retry_block | `PASS` | candidate retry-block lane/seed set matches the frozen benchmark |
| artifacts.duplicates | `PASS` | artifact roles are unique |
| artifacts | `PASS` | candidate pack exposes all required artifact roles |
| artifact_hash::candidate_metrics_json | `PASS` | candidate artifact `outputs/reports/historical_candidate_packs/multi_expert_hardening/artifacts/candidate_metrics.json` hash matches `865c0e2073d08c0a5b00865d6d0471cfe4b0b2faf05d6522630a206638ce9dc9` |
| artifact_hash::candidate_summary_markdown | `PASS` | candidate artifact `outputs/reports/historical_candidate_packs/multi_expert_hardening/artifacts/candidate_summary.md` hash matches `8ee7046ff8631833b456989cf7345fa5523df5ed0ac05a52a6dbe8d74317f75c` |
| artifact_hash::combined_report_csv | `PASS` | candidate artifact `outputs/reports/historical_candidate_packs/multi_expert_hardening/artifacts/combined_report.csv` hash matches `0014cb41569e11f8e1a7051697df64b14dd83e4de265e2d8dda6fcd42553f87c` |
| artifact_hash::combined_report_markdown | `PASS` | candidate artifact `outputs/reports/historical_candidate_packs/multi_expert_hardening/artifacts/combined_report.md` hash matches `c007dd24bda8f58b01150b7916007fc9b8e17f78003da4817a1304bed2909370` |
| artifact_hash::retry_block_report_csv | `PASS` | candidate artifact `outputs/reports/historical_candidate_packs/multi_expert_hardening/artifacts/retry_block_report.csv` hash matches `5f69b03e93a292c4e8b98317069100abe8c9b1d2ff948e44363ed80cc525c25a` |
| artifact_hash::retry_block_report_markdown | `PASS` | candidate artifact `outputs/reports/historical_candidate_packs/multi_expert_hardening/artifacts/retry_block_report.md` hash matches `62bc5550e3ebc13ebb8c55cd6acf025259ada5f3093d44c7ed02dd829592dac5` |
| artifact_consistency::candidate_metrics_json | `PASS` | candidate pack matches the candidate_metrics_json artifact on task, evaluation, controls, metrics, and actual sets |
| provenance.git_commit | `PASS` | candidate provenance git_commit `8d691d69559de8e2aba18481bebcb7151ac15d84` is well formed |
| provenance.git_dirty | `PASS` | candidate provenance git_dirty is `True` |

## Claim Gate Checks

| Check | Result | Detail |
| --- | --- | --- |
| frozen_pack::pack_type | `PASS` | pack type is `frozen_benchmark_pack` |
| frozen_pack::provenance.sealed_source_commit | `PASS` | sealed source commit `8d691d69559de8e2aba18481bebcb7151ac15d84` is well formed |
| frozen_pack::provenance.sealed_source_dirty | `PASS` | sealed source dirty flag is `True` |
| frozen_pack::manifest_hash | `PASS` | manifest hash matches `d10db1cff6f8cb55f434d3d11f0079d8f7e72ea45d2c46adc1c03909dece0bd6` |
| frozen_pack::schema_version | `PASS` | pack schema version `1` is recognized |
| frozen_pack::claim | `PASS` | sealed `claim` matches the manifest |
| frozen_pack::canonical_method | `PASS` | sealed `canonical_method` matches the manifest |
| frozen_pack::evaluation | `PASS` | sealed `evaluation` matches the manifest |
| frozen_pack::seed_groups | `PASS` | sealed `seed_groups` matches the manifest |
| frozen_pack::variants | `PASS` | sealed `variants` matches the manifest |
| frozen_pack::thresholds | `PASS` | sealed `thresholds` matches the manifest |
| frozen_pack::thaw_gate | `PASS` | sealed `thaw_gate` matches the manifest |
| frozen_pack::authoritative_artifacts | `PASS` | all required artifact keys are present |
| frozen_pack::artifact_hash::claim_gate_dry_run | `PASS` | artifact `outputs/reports/claim_gate_dry_run.md` hash matches `c38ee9a1df0197f405ed6aa16ad90a27f4d6b8f3d9f960588c0e3bbd70cf8dd7` |
| frozen_pack::artifact_hash::claim_gate_dry_run_json | `PASS` | artifact `outputs/reports/claim_gate_dry_run.json` hash matches `152056ccf996def48d342cfb6f98cdea2e30eae915a3333fdcc1721d8db7f41d` |
| frozen_pack::artifact_hash::claim_ledger | `PASS` | artifact `outputs/reports/claim_ledger.md` hash matches `8127d6aeb0b06bd69d13dbb75b37f991abebe4ba501fc60cf4a8d0c6a99d691d` |
| frozen_pack::artifact_hash::combined_doorkey_csv | `PASS` | artifact `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.csv` hash matches `f0c83d833ef9bfe3fa9e2db97f10e5bc42f6c1ab42805c14b21aa53ca0744b3b` |
| frozen_pack::artifact_hash::combined_doorkey_report | `PASS` | artifact `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.md` hash matches `e0cdd2b799538aad926156a19bdf88da19cb8c62a091ab16799a3e94cc87e19d` |
| frozen_pack::artifact_hash::final_block_csv | `PASS` | artifact `outputs/reports/lss_final_block_single_expert_control_report.csv` hash matches `8a7c5491529286337be0a727d2a71649ca4804b0e33efe8eba497354f46f7f2c` |
| frozen_pack::artifact_hash::final_block_report | `PASS` | artifact `outputs/reports/lss_final_block_single_expert_control_report.md` hash matches `084c6b96a5dfac5536af36009fe08664a42b21a61b230a3a03aa5e48a772812f` |
| frozen_pack::artifact_hash::forensic_casebook | `PASS` | artifact `outputs/reports/lss_forensic_casebook.md` hash matches `13d20ae9bfbf0def1595037882c44085c868a10b62c2a9e245e589d14de5f98e` |
| frozen_pack::artifact_hash::forensic_decision_memo | `PASS` | artifact `outputs/reports/lss_forensic_atlas_decision_memo.md` hash matches `646d16c2050c200301c9508dd1b579d39607aabc983913ffedc3a7a01b121998` |
| frozen_pack::artifact_hash::forensic_round_audit | `PASS` | artifact `outputs/reports/lss_forensic_round_audit.md` hash matches `cb9cb1a8fee052cca94fb6bfb078e1a01f6e7e286585dbea9337a27fb2c4505b` |
| frozen_pack::artifact_hash::forensic_route_locality | `PASS` | artifact `outputs/reports/lss_forensic_route_locality.md` hash matches `30ae43a28d514956eebac92c9af81490e04f8c4e1e64b6240ce44f9f1baf026c` |
| frozen_pack::artifact_hash::freeze_hardening_decision_memo | `PASS` | artifact `outputs/reports/freeze_hardening_decision_memo.md` hash matches `da0fdc73426b31c2a042bc71cef4a30307364c7c3333d4a3867b0b0b843bc7a5` |
| frozen_pack::artifact_hash::frozen_claim_envelope | `PASS` | artifact `outputs/reports/frozen_claim_envelope.md` hash matches `58cd74802397935f8abc2dbaf1119fa59046c3e15bc352deac098ace03e2a9d8` |
| frozen_pack::artifact_hash::frozen_validation_csv | `PASS` | artifact `outputs/reports/frozen_baseline_validation.csv` hash matches `3ec73864f9ab0d2cca0b115f9bee267ffd1e448231a15f70eebd8e3a7c03f162` |
| frozen_pack::artifact_hash::frozen_validation_json | `PASS` | artifact `outputs/reports/frozen_baseline_validation.json` hash matches `1888fb7ac3e78dead22811d99b88ee1d6b250c37de8fa0600e4bc7ea12a1918b` |
| frozen_pack::artifact_hash::frozen_validation_report | `PASS` | artifact `outputs/reports/frozen_baseline_validation.md` hash matches `9ce8e7f5055663abea1ea7444bde72ecc3789b9605a644ce24375f0149bf1248` |
| frozen_pack::artifact_hash::future_retry_template | `PASS` | artifact `outputs/reports/future_retry_template.md` hash matches `9ecaa4ac44e4ced4025bbb90959f3158dd3dbecab8b834b618f0e7270eb6e84e` |
| frozen_pack::artifact_hash::keycorridor_transfer_csv | `PASS` | artifact `outputs/reports/lss_keycorridor_transfer_report.csv` hash matches `fbc84ede909faa0a7170390451f09ada21a94e674d8b7ee1643effdb14468538` |
| frozen_pack::artifact_hash::keycorridor_transfer_report | `PASS` | artifact `outputs/reports/lss_keycorridor_transfer_report.md` hash matches `ee81af42d6e12ee8167c84a531db219627bdc8622913554c79d418be93ac93f9` |
| frozen_pack::artifact_hash::manifest_report | `PASS` | artifact `outputs/reports/frozen_claim_manifest_report.md` hash matches `a47eb42e06e4c8ba02e773c099e7a07917a071d2d0511c7aa325f2cc871947b6` |
| frozen_pack::artifact_hash::resume_scorecard | `PASS` | artifact `outputs/reports/lss_resume_qualification_scorecard.md` hash matches `843470eb82a25313b09cc796d3ef5f01f9efa6b02dfaf96aaa61e1bc80265028` |
| candidate_pack::pack_type | `PASS` | pack type is `candidate_result_pack` |
| candidate_pack::schema_version | `PASS` | candidate schema version `1` is recognized |
| candidate_pack::candidate_name | `PASS` | candidate name is `multi_expert_hardening` |
| candidate_pack::frozen_pack_reference.path | `PASS` | candidate references `outputs/reports/frozen_benchmark_pack.json` |
| candidate_pack::frozen_pack_reference.sha256 | `PASS` | candidate references frozen-pack hash `21f69b44b2a1567f6432e0ec2a0f5dce2c4f74ea256196d865fbdf384580a873` |
| candidate_pack::frozen_pack_reference.claim_id | `PASS` | candidate claim id matches `doorkey_frozen_claim` |
| candidate_pack::task | `PASS` | candidate task matches `DoorKey` |
| candidate_pack::evaluation | `PASS` | candidate uses the required DoorKey external-64 evaluation path |
| candidate_pack::requested_claims | `PASS` | requested claim keys are well formed |
| candidate_pack::controls_present.type | `PASS` | controls_present is a list of strings |
| candidate_pack::controls_present | `PASS` | candidate includes all required fairness controls |
| candidate_pack::metrics.combined.type | `PASS` | combined metrics are a mapping |
| candidate_pack::metrics.retry_block.type | `PASS` | retry-block metrics are a mapping |
| candidate_pack::metrics.combined | `PASS` | combined metrics expose every required variant field |
| candidate_pack::metrics.retry_block | `INCONCLUSIVE` | retry-block metrics are missing fields: `kl_lss_single_expert:<variant>` |
| candidate_pack::actual_sets.combined | `PASS` | candidate combined lane/seed set matches the frozen benchmark |
| candidate_pack::actual_sets.retry_block | `PASS` | candidate retry-block lane/seed set matches the frozen benchmark |
| candidate_pack::artifacts.duplicates | `PASS` | artifact roles are unique |
| candidate_pack::artifacts | `PASS` | candidate pack exposes all required artifact roles |
| candidate_pack::artifact_hash::candidate_metrics_json | `PASS` | candidate artifact `outputs/reports/historical_candidate_packs/multi_expert_hardening/artifacts/candidate_metrics.json` hash matches `865c0e2073d08c0a5b00865d6d0471cfe4b0b2faf05d6522630a206638ce9dc9` |
| candidate_pack::artifact_hash::candidate_summary_markdown | `PASS` | candidate artifact `outputs/reports/historical_candidate_packs/multi_expert_hardening/artifacts/candidate_summary.md` hash matches `8ee7046ff8631833b456989cf7345fa5523df5ed0ac05a52a6dbe8d74317f75c` |
| candidate_pack::artifact_hash::combined_report_csv | `PASS` | candidate artifact `outputs/reports/historical_candidate_packs/multi_expert_hardening/artifacts/combined_report.csv` hash matches `0014cb41569e11f8e1a7051697df64b14dd83e4de265e2d8dda6fcd42553f87c` |
| candidate_pack::artifact_hash::combined_report_markdown | `PASS` | candidate artifact `outputs/reports/historical_candidate_packs/multi_expert_hardening/artifacts/combined_report.md` hash matches `c007dd24bda8f58b01150b7916007fc9b8e17f78003da4817a1304bed2909370` |
| candidate_pack::artifact_hash::retry_block_report_csv | `PASS` | candidate artifact `outputs/reports/historical_candidate_packs/multi_expert_hardening/artifacts/retry_block_report.csv` hash matches `5f69b03e93a292c4e8b98317069100abe8c9b1d2ff948e44363ed80cc525c25a` |
| candidate_pack::artifact_hash::retry_block_report_markdown | `PASS` | candidate artifact `outputs/reports/historical_candidate_packs/multi_expert_hardening/artifacts/retry_block_report.md` hash matches `62bc5550e3ebc13ebb8c55cd6acf025259ada5f3093d44c7ed02dd829592dac5` |
| candidate_pack::artifact_consistency::candidate_metrics_json | `PASS` | candidate pack matches the candidate_metrics_json artifact on task, evaluation, controls, metrics, and actual sets |
| candidate_pack::provenance.git_commit | `PASS` | candidate provenance git_commit `8d691d69559de8e2aba18481bebcb7151ac15d84` is well formed |
| candidate_pack::provenance.git_dirty | `PASS` | candidate provenance git_dirty is `True` |

## Verdict

INCONCLUSIVE: missing prerequisites
