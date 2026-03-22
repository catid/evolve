# Frozen Benchmark Pack Validation

- pack: `outputs/reports/round6_current_benchmark_pack.json`

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

## Verdict

PASS: frozen benchmark pack validated
