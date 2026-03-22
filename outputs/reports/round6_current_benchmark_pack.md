# Frozen Benchmark Pack

- pack json: `outputs/reports/round6_current_benchmark_pack.json`
- claim id: `doorkey_round6_current_claim`
- frozen status: `current_active_benchmark`
- sealed source commit: `251c92bc33d7b9ec5aa313dcbdd1a9c245bfa7fe`
- sealed source dirty: `True`
- schema version: `1`
- manifest: `configs/claims/doorkey_round6_current_claim.yaml`

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
| `state_reconciliation` | `outputs/reports/next_round_state_reconciliation.md` | `bb5148f3b98a24f5784f3d426b7fb2c3bbb3a81f4c1bcc0bcbe546a51fb6c388` | 1714 |
| `baseline_sync` | `outputs/reports/next_round_baseline_sync.md` | `8ebcf0635bcb86185bc00c6daeca5183642d6a9e7e7c2963cbb1ebcf72371bd7` | 1219 |
| `claim_ledger_snapshot` | `outputs/reports/claim_ledger_round6_current.md` | `ee26c73a78b1ca070a13df98bc508e3e6480dc17b9bae3ce0ab026ce76a8100d` | 13115 |
| `next_mega_decision_memo` | `outputs/reports/next_mega_portfolio_decision_memo.md` | `4a9c3d76e31a6906531ca0a75963a64383dfd9b8eb1b9eab23228e9409731597` | 827 |
| `next_mega_gate_report` | `outputs/reports/next_mega_portfolio_gate_report.md` | `d107e363f89d91be59ba5ad378e51b42d1a337badfdc9b0a103549fe26db862d` | 18143 |
| `next_mega_stage1_fruitful` | `outputs/reports/next_mega_portfolio_stage1_screening_fruitful.md` | `64e7cbe3ac8283bb8ecb88644eaac0d5bb48e763c264096af6057aeffba03914` | 53166 |
| `next_mega_stage1_exploratory` | `outputs/reports/next_mega_portfolio_stage1_screening_exploratory.md` | `970654a48f3d5aa20ad7c1d2cad713a0e5d7760cd769b32060fcf2a1963efecc` | 55243 |
| `next_mega_holdout` | `outputs/reports/next_mega_portfolio_stage4_holdout.md` | `a30ef1d2a2ea9d6ad445c9548c52f9d640b9f355b796a87a8faefe8405e42f64` | 3226 |
| `next_mega_antiregression` | `outputs/reports/next_mega_portfolio_stage5_antiregression.md` | `46fa6a07dcef3bc20383b03e45357c71c2ba532491d78e013f2a737a529f1982` | 3081 |
| `next_mega_route_validation` | `outputs/reports/next_mega_portfolio_stage6_route_validation.md` | `c662b19e17d1c3effc23058b77efb0c4765878ee6a5648e03b96ad7455371d92` | 661 |
| `next_mega_stability` | `outputs/reports/next_mega_portfolio_stage7_stability.md` | `615d6461dfc978a4d8d8a3e438f59079e30276f7a5044cc7dd7d75ab852efd2f` | 518 |

## Provenance

- This pack is the immutable machine-readable baseline unit for the current frozen DoorKey claim.
- It is derived from the frozen manifest and authoritative reports, and it is meant to be validated before any future thaw candidate is discussed.
