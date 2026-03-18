# Frozen Benchmark Pack

- pack json: `outputs/reports/frozen_benchmark_pack.json`
- claim id: `doorkey_frozen_claim`
- frozen status: `frozen`
- sealed source commit: `90f4ca1e3b9a572156e49d4af86d273a748cea43`
- sealed source dirty: `True`
- schema version: `1`
- manifest: `configs/claims/doorkey_frozen_claim.yaml`

## Claim Envelope

- allowed: `bounded teacher-guided DoorKey SARE result`
- disallowed:
  - `ppo_only_routed_win`
  - `specifically_multi_expert_routed_advantage`
  - `cross_task_routed_advantage`
  - `keycorridor_transfer_claim`

## Frozen Thresholds

- combined KL learner-state `SARE` mean: `0.7122`
- retry-block KL learner-state `SARE` mean: `0.3125`
- retry-block KL learner-state `single_expert` mean: `0.4635`

## Canonical Artifacts

| Key | Path | SHA256 | Size (bytes) |
| --- | --- | --- | ---: |
| `frozen_claim_envelope` | `outputs/reports/frozen_claim_envelope.md` | `58cd74802397935f8abc2dbaf1119fa59046c3e15bc352deac098ace03e2a9d8` | 1083 |
| `manifest_report` | `outputs/reports/frozen_claim_manifest_report.md` | `a47eb42e06e4c8ba02e773c099e7a07917a071d2d0511c7aa325f2cc871947b6` | 3157 |
| `frozen_validation_report` | `outputs/reports/frozen_baseline_validation.md` | `9ce8e7f5055663abea1ea7444bde72ecc3789b9605a644ce24375f0149bf1248` | 3115 |
| `frozen_validation_csv` | `outputs/reports/frozen_baseline_validation.csv` | `3ec73864f9ab0d2cca0b115f9bee267ffd1e448231a15f70eebd8e3a7c03f162` | 2046 |
| `frozen_validation_json` | `outputs/reports/frozen_baseline_validation.json` | `1888fb7ac3e78dead22811d99b88ee1d6b250c37de8fa0600e4bc7ea12a1918b` | 9679 |
| `claim_gate_dry_run` | `outputs/reports/claim_gate_dry_run.md` | `c38ee9a1df0197f405ed6aa16ad90a27f4d6b8f3d9f960588c0e3bbd70cf8dd7` | 1700 |
| `claim_gate_dry_run_json` | `outputs/reports/claim_gate_dry_run.json` | `152056ccf996def48d342cfb6f98cdea2e30eae915a3333fdcc1721d8db7f41d` | 2556 |
| `claim_ledger` | `outputs/reports/claim_ledger.md` | `b6e21c2b86d6ef74de6f27c986fc566de5652a4d08512094367809c579992f41` | 5371 |
| `future_retry_template` | `outputs/reports/future_retry_template.md` | `9ecaa4ac44e4ced4025bbb90959f3158dd3dbecab8b834b618f0e7270eb6e84e` | 1555 |
| `freeze_hardening_decision_memo` | `outputs/reports/freeze_hardening_decision_memo.md` | `da0fdc73426b31c2a042bc71cef4a30307364c7c3333d4a3867b0b0b843bc7a5` | 1474 |
| `combined_doorkey_report` | `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.md` | `e0cdd2b799538aad926156a19bdf88da19cb8c62a091ab16799a3e94cc87e19d` | 1687 |
| `combined_doorkey_csv` | `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.csv` | `f0c83d833ef9bfe3fa9e2db97f10e5bc42f6c1ab42805c14b21aa53ca0744b3b` | 182578 |
| `final_block_report` | `outputs/reports/lss_final_block_single_expert_control_report.md` | `084c6b96a5dfac5536af36009fe08664a42b21a61b230a3a03aa5e48a772812f` | 950 |
| `final_block_csv` | `outputs/reports/lss_final_block_single_expert_control_report.csv` | `8a7c5491529286337be0a727d2a71649ca4804b0e33efe8eba497354f46f7f2c` | 47509 |
| `forensic_casebook` | `outputs/reports/lss_forensic_casebook.md` | `13d20ae9bfbf0def1595037882c44085c868a10b62c2a9e245e589d14de5f98e` | 12695 |
| `forensic_round_audit` | `outputs/reports/lss_forensic_round_audit.md` | `cb9cb1a8fee052cca94fb6bfb078e1a01f6e7e286585dbea9337a27fb2c4505b` | 4282 |
| `forensic_route_locality` | `outputs/reports/lss_forensic_route_locality.md` | `30ae43a28d514956eebac92c9af81490e04f8c4e1e64b6240ce44f9f1baf026c` | 5435 |
| `forensic_decision_memo` | `outputs/reports/lss_forensic_atlas_decision_memo.md` | `646d16c2050c200301c9508dd1b579d39607aabc983913ffedc3a7a01b121998` | 1189 |
| `resume_scorecard` | `outputs/reports/lss_resume_qualification_scorecard.md` | `843470eb82a25313b09cc796d3ef5f01f9efa6b02dfaf96aaa61e1bc80265028` | 2629 |
| `keycorridor_transfer_report` | `outputs/reports/lss_keycorridor_transfer_report.md` | `ee81af42d6e12ee8167c84a531db219627bdc8622913554c79d418be93ac93f9` | 440 |
| `keycorridor_transfer_csv` | `outputs/reports/lss_keycorridor_transfer_report.csv` | `fbc84ede909faa0a7170390451f09ada21a94e674d8b7ee1643effdb14468538` | 32199 |

## Provenance

- This pack is the immutable machine-readable baseline unit for the current frozen DoorKey claim.
- It is derived from the frozen manifest and authoritative reports, and it is meant to be validated before any future thaw candidate is discussed.
