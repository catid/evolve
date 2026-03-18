# Frozen Benchmark Pack Validation

- pack: `outputs/reports/frozen_benchmark_pack.json`

| Check | Result | Detail |
| --- | --- | --- |
| pack_type | `PASS` | pack type is `frozen_benchmark_pack` |
| provenance.sealed_source_commit | `PASS` | sealed source commit `21a890285831feb89c6de03ea16619239ff732e3` is well formed |
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
| artifact_hash::claim_ledger | `PASS` | artifact `outputs/reports/claim_ledger.md` hash matches `4145f1f38643a7742410eeedf48713123a10cf225eef0cf4dfe80f8e95b9054e` |
| artifact_hash::combined_doorkey_csv | `PASS` | artifact `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.csv` hash matches `f0c83d833ef9bfe3fa9e2db97f10e5bc42f6c1ab42805c14b21aa53ca0744b3b` |
| artifact_hash::combined_doorkey_report | `PASS` | artifact `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.md` hash matches `e0cdd2b799538aad926156a19bdf88da19cb8c62a091ab16799a3e94cc87e19d` |
| artifact_hash::final_block_csv | `PASS` | artifact `outputs/reports/lss_final_block_single_expert_control_report.csv` hash matches `8a7c5491529286337be0a727d2a71649ca4804b0e33efe8eba497354f46f7f2c` |
| artifact_hash::final_block_report | `PASS` | artifact `outputs/reports/lss_final_block_single_expert_control_report.md` hash matches `084c6b96a5dfac5536af36009fe08664a42b21a61b230a3a03aa5e48a772812f` |
| artifact_hash::forensic_casebook | `PASS` | artifact `outputs/reports/lss_forensic_casebook.md` hash matches `13d20ae9bfbf0def1595037882c44085c868a10b62c2a9e245e589d14de5f98e` |
| artifact_hash::forensic_decision_memo | `PASS` | artifact `outputs/reports/lss_forensic_atlas_decision_memo.md` hash matches `646d16c2050c200301c9508dd1b579d39607aabc983913ffedc3a7a01b121998` |
| artifact_hash::forensic_round_audit | `PASS` | artifact `outputs/reports/lss_forensic_round_audit.md` hash matches `cb9cb1a8fee052cca94fb6bfb078e1a01f6e7e286585dbea9337a27fb2c4505b` |
| artifact_hash::forensic_route_locality | `PASS` | artifact `outputs/reports/lss_forensic_route_locality.md` hash matches `30ae43a28d514956eebac92c9af81490e04f8c4e1e64b6240ce44f9f1baf026c` |
| artifact_hash::freeze_hardening_decision_memo | `PASS` | artifact `outputs/reports/freeze_hardening_decision_memo.md` hash matches `da0fdc73426b31c2a042bc71cef4a30307364c7c3333d4a3867b0b0b843bc7a5` |
| artifact_hash::frozen_claim_envelope | `PASS` | artifact `outputs/reports/frozen_claim_envelope.md` hash matches `566d777b34a002a98719e2e8f56d2e20c61868b15f6151fa52115125048e8dcf` |
| artifact_hash::frozen_validation_csv | `PASS` | artifact `outputs/reports/frozen_baseline_validation.csv` hash matches `3ec73864f9ab0d2cca0b115f9bee267ffd1e448231a15f70eebd8e3a7c03f162` |
| artifact_hash::frozen_validation_json | `PASS` | artifact `outputs/reports/frozen_baseline_validation.json` hash matches `f8aa55424502e42c03bdd20eaa920ad24f4fdc20a4105bd92620e7b6a3498ba9` |
| artifact_hash::frozen_validation_report | `PASS` | artifact `outputs/reports/frozen_baseline_validation.md` hash matches `f778f060dded331c9eec1e7b22dea04420698494249218a3020e466e42dd23df` |
| artifact_hash::future_retry_template | `PASS` | artifact `outputs/reports/future_retry_template.md` hash matches `9ecaa4ac44e4ced4025bbb90959f3158dd3dbecab8b834b618f0e7270eb6e84e` |
| artifact_hash::keycorridor_transfer_csv | `PASS` | artifact `outputs/reports/lss_keycorridor_transfer_report.csv` hash matches `fbc84ede909faa0a7170390451f09ada21a94e674d8b7ee1643effdb14468538` |
| artifact_hash::keycorridor_transfer_report | `PASS` | artifact `outputs/reports/lss_keycorridor_transfer_report.md` hash matches `ee81af42d6e12ee8167c84a531db219627bdc8622913554c79d418be93ac93f9` |
| artifact_hash::manifest_report | `PASS` | artifact `outputs/reports/frozen_claim_manifest_report.md` hash matches `a47eb42e06e4c8ba02e773c099e7a07917a071d2d0511c7aa325f2cc871947b6` |
| artifact_hash::resume_scorecard | `PASS` | artifact `outputs/reports/lss_resume_qualification_scorecard.md` hash matches `843470eb82a25313b09cc796d3ef5f01f9efa6b02dfaf96aaa61e1bc80265028` |

## Verdict

PASS: frozen benchmark pack validated
