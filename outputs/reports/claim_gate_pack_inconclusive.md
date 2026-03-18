# Pack-Based Claim Gate Dry Run

- frozen benchmark pack: `outputs/reports/frozen_benchmark_pack.json`
- candidate result pack: `outputs/reports/candidate_result_pack_incomplete.json`

## Frozen Pack Validation

| Check | Result | Detail |
| --- | --- | --- |
| pack_type | `PASS` | pack type is `frozen_benchmark_pack` |
| provenance.sealed_source_commit | `PASS` | sealed source commit `91fab1f7a3f91402fcca730ad15a9185aac92cf1` is well formed |
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
| artifact_hash::claim_ledger | `PASS` | artifact `outputs/reports/claim_ledger.md` hash matches `26ec0b1ffff4d5805a5e1c93cd1ea813b5a930d0fb85c665f140dd250622bf9c` |
| artifact_hash::combined_doorkey_csv | `PASS` | artifact `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.csv` hash matches `f0c83d833ef9bfe3fa9e2db97f10e5bc42f6c1ab42805c14b21aa53ca0744b3b` |
| artifact_hash::combined_doorkey_report | `PASS` | artifact `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.md` hash matches `e0cdd2b799538aad926156a19bdf88da19cb8c62a091ab16799a3e94cc87e19d` |
| artifact_hash::final_block_csv | `PASS` | artifact `outputs/reports/lss_final_block_single_expert_control_report.csv` hash matches `8a7c5491529286337be0a727d2a71649ca4804b0e33efe8eba497354f46f7f2c` |
| artifact_hash::final_block_report | `PASS` | artifact `outputs/reports/lss_final_block_single_expert_control_report.md` hash matches `084c6b96a5dfac5536af36009fe08664a42b21a61b230a3a03aa5e48a772812f` |
| artifact_hash::forensic_casebook | `PASS` | artifact `outputs/reports/lss_forensic_casebook.md` hash matches `13d20ae9bfbf0def1595037882c44085c868a10b62c2a9e245e589d14de5f98e` |
| artifact_hash::forensic_decision_memo | `PASS` | artifact `outputs/reports/lss_forensic_atlas_decision_memo.md` hash matches `646d16c2050c200301c9508dd1b579d39607aabc983913ffedc3a7a01b121998` |
| artifact_hash::forensic_round_audit | `PASS` | artifact `outputs/reports/lss_forensic_round_audit.md` hash matches `cb9cb1a8fee052cca94fb6bfb078e1a01f6e7e286585dbea9337a27fb2c4505b` |
| artifact_hash::forensic_route_locality | `PASS` | artifact `outputs/reports/lss_forensic_route_locality.md` hash matches `30ae43a28d514956eebac92c9af81490e04f8c4e1e64b6240ce44f9f1baf026c` |
| artifact_hash::freeze_hardening_decision_memo | `PASS` | artifact `outputs/reports/freeze_hardening_decision_memo.md` hash matches `da0fdc73426b31c2a042bc71cef4a30307364c7c3333d4a3867b0b0b843bc7a5` |
| artifact_hash::frozen_claim_envelope | `PASS` | artifact `outputs/reports/frozen_claim_envelope.md` hash matches `cead2e3f00e6f18b2cd88d41abe0cfe3cab44bb8936822ff6a3ab03aa03c43a9` |
| artifact_hash::frozen_validation_csv | `PASS` | artifact `outputs/reports/frozen_baseline_validation.csv` hash matches `3ec73864f9ab0d2cca0b115f9bee267ffd1e448231a15f70eebd8e3a7c03f162` |
| artifact_hash::frozen_validation_json | `PASS` | artifact `outputs/reports/frozen_baseline_validation.json` hash matches `250b556bd800c8c9d0225a4b6c25c2ce712b19e01d6db4397bdaecdc6303a1d5` |
| artifact_hash::frozen_validation_report | `PASS` | artifact `outputs/reports/frozen_baseline_validation.md` hash matches `532a1f08bdd313d84acf1a57c97916b3024789df967c0286587b88def0b490c2` |
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
| candidate_name | `PASS` | candidate name is `incomplete_candidate_pack_example` |
| frozen_pack_reference.path | `PASS` | candidate references `outputs/reports/frozen_benchmark_pack.json` |
| frozen_pack_reference.sha256 | `PASS` | candidate references frozen-pack hash `8e36f4623ad584549f0b3a4be7bef9fbec62053704cef6672b6bb28a2608f205` |
| frozen_pack_reference.claim_id | `PASS` | candidate claim id matches `doorkey_frozen_claim` |
| task | `PASS` | candidate task matches `DoorKey` |
| evaluation | `PASS` | candidate uses the required DoorKey external-64 evaluation path |
| requested_claims | `PASS` | requested claim keys are well formed |
| controls_present.type | `PASS` | controls_present is a list of strings |
| controls_present | `FAIL` | candidate is missing required controls: `baseline_sare`, `kl_lss_single_expert`, `kl_lss_token_dense` |
| metrics.combined.type | `PASS` | combined metrics are a mapping |
| metrics.retry_block.type | `FAIL` | retry-block metrics must be a mapping |
| metrics.combined | `FAIL` | combined metrics are missing fields: `baseline_sare:<variant>`, `kl_lss_single_expert:<variant>`, `kl_lss_token_dense:<variant>`, `recovered_token_dense:<variant>` |
| metrics.retry_block | `FAIL` | retry-block metrics are missing fields: `<retry-block-mapping>` |
| actual_sets.combined | `FAIL` | candidate combined lane/seed set `[]` does not match frozen `[('fresh', 23), ('fresh', 29), ('fresh', 31), ('fresh_extra', 37), ('fresh_extra', 41), ('fresh_extra', 43), ('fresh_final', 47), ('fresh_final', 53), ('fresh_final', 59), ('original', 7), ('original', 11), ('original', 19)]` |
| actual_sets.retry_block | `FAIL` | candidate retry-block lane/seed set `[]` does not match frozen `[('fresh_final', 47), ('fresh_final', 53), ('fresh_final', 59)]` |
| artifacts.duplicates | `PASS` | artifact roles are unique |
| artifacts | `FAIL` | candidate pack is missing required artifact roles: `candidate_summary_markdown`, `combined_report_csv`, `combined_report_markdown`, `retry_block_report_csv`, `retry_block_report_markdown` |
| artifact_hash::candidate_metrics_json | `FAIL` | candidate artifact `outputs/reports/does_not_exist.json` is missing |
| provenance.git_commit | `PASS` | candidate provenance git_commit `91fab1f7a3f91402fcca730ad15a9185aac92cf1` is well formed |
| provenance.git_dirty | `PASS` | candidate provenance git_dirty is `True` |

## Claim Gate Checks

| Check | Result | Detail |
| --- | --- | --- |
| frozen_pack::pack_type | `PASS` | pack type is `frozen_benchmark_pack` |
| frozen_pack::provenance.sealed_source_commit | `PASS` | sealed source commit `91fab1f7a3f91402fcca730ad15a9185aac92cf1` is well formed |
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
| frozen_pack::artifact_hash::claim_ledger | `PASS` | artifact `outputs/reports/claim_ledger.md` hash matches `26ec0b1ffff4d5805a5e1c93cd1ea813b5a930d0fb85c665f140dd250622bf9c` |
| frozen_pack::artifact_hash::combined_doorkey_csv | `PASS` | artifact `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.csv` hash matches `f0c83d833ef9bfe3fa9e2db97f10e5bc42f6c1ab42805c14b21aa53ca0744b3b` |
| frozen_pack::artifact_hash::combined_doorkey_report | `PASS` | artifact `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.md` hash matches `e0cdd2b799538aad926156a19bdf88da19cb8c62a091ab16799a3e94cc87e19d` |
| frozen_pack::artifact_hash::final_block_csv | `PASS` | artifact `outputs/reports/lss_final_block_single_expert_control_report.csv` hash matches `8a7c5491529286337be0a727d2a71649ca4804b0e33efe8eba497354f46f7f2c` |
| frozen_pack::artifact_hash::final_block_report | `PASS` | artifact `outputs/reports/lss_final_block_single_expert_control_report.md` hash matches `084c6b96a5dfac5536af36009fe08664a42b21a61b230a3a03aa5e48a772812f` |
| frozen_pack::artifact_hash::forensic_casebook | `PASS` | artifact `outputs/reports/lss_forensic_casebook.md` hash matches `13d20ae9bfbf0def1595037882c44085c868a10b62c2a9e245e589d14de5f98e` |
| frozen_pack::artifact_hash::forensic_decision_memo | `PASS` | artifact `outputs/reports/lss_forensic_atlas_decision_memo.md` hash matches `646d16c2050c200301c9508dd1b579d39607aabc983913ffedc3a7a01b121998` |
| frozen_pack::artifact_hash::forensic_round_audit | `PASS` | artifact `outputs/reports/lss_forensic_round_audit.md` hash matches `cb9cb1a8fee052cca94fb6bfb078e1a01f6e7e286585dbea9337a27fb2c4505b` |
| frozen_pack::artifact_hash::forensic_route_locality | `PASS` | artifact `outputs/reports/lss_forensic_route_locality.md` hash matches `30ae43a28d514956eebac92c9af81490e04f8c4e1e64b6240ce44f9f1baf026c` |
| frozen_pack::artifact_hash::freeze_hardening_decision_memo | `PASS` | artifact `outputs/reports/freeze_hardening_decision_memo.md` hash matches `da0fdc73426b31c2a042bc71cef4a30307364c7c3333d4a3867b0b0b843bc7a5` |
| frozen_pack::artifact_hash::frozen_claim_envelope | `PASS` | artifact `outputs/reports/frozen_claim_envelope.md` hash matches `cead2e3f00e6f18b2cd88d41abe0cfe3cab44bb8936822ff6a3ab03aa03c43a9` |
| frozen_pack::artifact_hash::frozen_validation_csv | `PASS` | artifact `outputs/reports/frozen_baseline_validation.csv` hash matches `3ec73864f9ab0d2cca0b115f9bee267ffd1e448231a15f70eebd8e3a7c03f162` |
| frozen_pack::artifact_hash::frozen_validation_json | `PASS` | artifact `outputs/reports/frozen_baseline_validation.json` hash matches `250b556bd800c8c9d0225a4b6c25c2ce712b19e01d6db4397bdaecdc6303a1d5` |
| frozen_pack::artifact_hash::frozen_validation_report | `PASS` | artifact `outputs/reports/frozen_baseline_validation.md` hash matches `532a1f08bdd313d84acf1a57c97916b3024789df967c0286587b88def0b490c2` |
| frozen_pack::artifact_hash::future_retry_template | `PASS` | artifact `outputs/reports/future_retry_template.md` hash matches `9ecaa4ac44e4ced4025bbb90959f3158dd3dbecab8b834b618f0e7270eb6e84e` |
| frozen_pack::artifact_hash::keycorridor_transfer_csv | `PASS` | artifact `outputs/reports/lss_keycorridor_transfer_report.csv` hash matches `fbc84ede909faa0a7170390451f09ada21a94e674d8b7ee1643effdb14468538` |
| frozen_pack::artifact_hash::keycorridor_transfer_report | `PASS` | artifact `outputs/reports/lss_keycorridor_transfer_report.md` hash matches `ee81af42d6e12ee8167c84a531db219627bdc8622913554c79d418be93ac93f9` |
| frozen_pack::artifact_hash::manifest_report | `PASS` | artifact `outputs/reports/frozen_claim_manifest_report.md` hash matches `a47eb42e06e4c8ba02e773c099e7a07917a071d2d0511c7aa325f2cc871947b6` |
| frozen_pack::artifact_hash::resume_scorecard | `PASS` | artifact `outputs/reports/lss_resume_qualification_scorecard.md` hash matches `843470eb82a25313b09cc796d3ef5f01f9efa6b02dfaf96aaa61e1bc80265028` |
| candidate_pack::pack_type | `PASS` | pack type is `candidate_result_pack` |
| candidate_pack::schema_version | `PASS` | candidate schema version `1` is recognized |
| candidate_pack::candidate_name | `PASS` | candidate name is `incomplete_candidate_pack_example` |
| candidate_pack::frozen_pack_reference.path | `PASS` | candidate references `outputs/reports/frozen_benchmark_pack.json` |
| candidate_pack::frozen_pack_reference.sha256 | `PASS` | candidate references frozen-pack hash `8e36f4623ad584549f0b3a4be7bef9fbec62053704cef6672b6bb28a2608f205` |
| candidate_pack::frozen_pack_reference.claim_id | `PASS` | candidate claim id matches `doorkey_frozen_claim` |
| candidate_pack::task | `PASS` | candidate task matches `DoorKey` |
| candidate_pack::evaluation | `PASS` | candidate uses the required DoorKey external-64 evaluation path |
| candidate_pack::requested_claims | `PASS` | requested claim keys are well formed |
| candidate_pack::controls_present.type | `PASS` | controls_present is a list of strings |
| candidate_pack::controls_present | `INCONCLUSIVE` | candidate is missing required controls: `baseline_sare`, `kl_lss_single_expert`, `kl_lss_token_dense` |
| candidate_pack::metrics.combined.type | `PASS` | combined metrics are a mapping |
| candidate_pack::metrics.retry_block.type | `INCONCLUSIVE` | retry-block metrics must be a mapping |
| candidate_pack::metrics.combined | `INCONCLUSIVE` | combined metrics are missing fields: `baseline_sare:<variant>`, `kl_lss_single_expert:<variant>`, `kl_lss_token_dense:<variant>`, `recovered_token_dense:<variant>` |
| candidate_pack::metrics.retry_block | `INCONCLUSIVE` | retry-block metrics are missing fields: `<retry-block-mapping>` |
| candidate_pack::actual_sets.combined | `INCONCLUSIVE` | candidate combined lane/seed set `[]` does not match frozen `[('fresh', 23), ('fresh', 29), ('fresh', 31), ('fresh_extra', 37), ('fresh_extra', 41), ('fresh_extra', 43), ('fresh_final', 47), ('fresh_final', 53), ('fresh_final', 59), ('original', 7), ('original', 11), ('original', 19)]` |
| candidate_pack::actual_sets.retry_block | `INCONCLUSIVE` | candidate retry-block lane/seed set `[]` does not match frozen `[('fresh_final', 47), ('fresh_final', 53), ('fresh_final', 59)]` |
| candidate_pack::artifacts.duplicates | `PASS` | artifact roles are unique |
| candidate_pack::artifacts | `INCONCLUSIVE` | candidate pack is missing required artifact roles: `candidate_summary_markdown`, `combined_report_csv`, `combined_report_markdown`, `retry_block_report_csv`, `retry_block_report_markdown` |
| candidate_pack::artifact_hash::candidate_metrics_json | `INCONCLUSIVE` | candidate artifact `outputs/reports/does_not_exist.json` is missing |
| candidate_pack::provenance.git_commit | `PASS` | candidate provenance git_commit `91fab1f7a3f91402fcca730ad15a9185aac92cf1` is well formed |
| candidate_pack::provenance.git_dirty | `PASS` | candidate provenance git_dirty is `True` |

## Verdict

INCONCLUSIVE: missing prerequisites
