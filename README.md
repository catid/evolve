# PSMN RL

Research code for Packet-Switched Morphogenic Network reinforcement learning experiments on small game environments.

The first milestone focuses on MiniGrid with PPO and DDP, comparing dense tokenized baselines against routed variants:

- flattened dense baseline
- tokenized dense baseline
- tokenized single-expert baseline
- SARE
- TREG-H
- SRW
- POR

`token_gru` is available as a diagnostic memory probe, but it is not treated as a fair mainline PPO control until sequence-aware rollout batching exists.

## Environment Setup

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -U pip setuptools wheel
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
uv pip install -e .
```

Optional dev tools:

```bash
uv pip install -e '.[dev]'
```

Initialize Beads at repo root:

```bash
bd init
./scripts/seed_beads.sh
```

## Quick Start

Single-process smoke run:

```bash
python -m psmn_rl.train --config configs/baseline/minigrid_dense.yaml --max-updates 1 --device cpu
```

Use all visible GPUs with `torchrun`:

```bash
NPROC_PER_NODE=$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m psmn_rl.launch \
  --config configs/baseline/minigrid_dense.yaml
```

Run SARE on MiniGrid:

```bash
./scripts/train_minigrid.sh configs/sare/minigrid_doorkey.yaml
```

Procgen is wired as an optional suite, but on Python 3.12 it currently depends on an external Gymnasium-compatible Procgen port. If you try a Procgen config without that package installed, the code raises a clear compatibility error instead of failing later in training.

Install the pinned Procgen port used for this repo:

```bash
./scripts/install_procgen_port.sh
```

Then a Procgen smoke run works with the same trainer:

```bash
python -m psmn_rl.train --config configs/baseline/procgen_coinrun_dense.yaml --max-updates 1 --device cpu
```

Evaluate a checkpoint:

```bash
./scripts/eval.sh outputs/minigrid_dense/latest.pt configs/baseline/minigrid_dense.yaml
```

Run the MiniGrid ablation sweep:

```bash
PSMN_MAX_UPDATES=1 ./scripts/run_minigrid_ablations.sh
```

This writes per-variant outputs plus a markdown summary under `outputs/ablations/`.

Run the 4-environment MiniGrid control-vs-SARE suite:

```bash
PSMN_MAX_UPDATES=2 ./scripts/run_minigrid_suite_ablations.sh
```

This produces a multi-environment summary and grouped comparison report under `outputs/ablations/minigrid_suite_ddp/`.

## Control-First MiniGrid Harness

The active experiment flow is now control-first rather than architecture-first.

- `configs/baseline/`: original short-run smoke/falsification configs
- `configs/diagnostic/`: tiny overfit, sanity-tier, fully observed, and memory-probe diagnostics
- `configs/experiments/`: longer-run control baselines and fair routed reruns

Key scripts:

```bash
./scripts/run_tiny_overfit.sh
./scripts/run_minigrid_sanity_suite.sh
./scripts/run_minigrid_baseline_suite.sh
./scripts/run_sare_comparison_sweep.sh
./scripts/run_policy_extraction_diagnostics.sh
./scripts/run_tokenization_gap_diagnostics.sh
./scripts/run_token_control_recovery_sweep.sh
./scripts/run_sare_fair_retest.sh
./scripts/eval_policy_modes.sh
./scripts/run_lss_claim_consolidation_reproduction.sh
./scripts/run_lss_claim_consolidation_fresh_matched_controls.sh
./scripts/run_lss_claim_consolidation_route_dependence.sh
./scripts/run_lss_claim_broadening_reproduction.sh
./scripts/run_lss_claim_broadening_single_expert_controls.sh
./scripts/run_lss_claim_broadening_extended_route_dependence.sh
./scripts/run_lss_claim_broadening_additional_fresh_block.sh
./scripts/run_lss_multi_expert_hardening_reproduction.sh
./scripts/run_lss_multi_expert_hardening_fresh_single_expert_controls.sh
./scripts/run_lss_multi_expert_hardening_seed29_forensics.sh
./scripts/run_lss_multi_expert_hardening_broader_route_dependence.sh
./scripts/run_lss_multi_expert_hardening_final_fresh_block.sh
./scripts/run_lss_multi_expert_hardening_finalize.sh
./scripts/run_frozen_baseline_validation.sh
./scripts/run_frozen_benchmark_pack_validation.sh
./scripts/run_claim_gate.sh
./scripts/run_claim_gate_conformance.sh
./scripts/run_claim_history_replay.sh
./scripts/run_claim_gate_redteam_finalize.sh
./scripts/run_freeze_hardening_finalize.sh
./scripts/run_lss_long_campaign_register.sh
./scripts/run_lss_long_campaign_stage2.sh
./scripts/run_lss_long_campaign_stage3.sh
./scripts/run_lss_long_campaign_stage4.sh
./scripts/run_lss_long_campaign_stage5.sh
./scripts/run_lss_long_campaign_stage6.sh
./scripts/run_lss_long_campaign_finalize.sh
./scripts/run_lss_post_pass_campaign_register.sh
./scripts/run_lss_post_pass_campaign_stage1.sh
./scripts/run_lss_post_pass_campaign_stage2.sh
./scripts/run_lss_post_pass_campaign_stage3.sh
./scripts/run_lss_post_pass_campaign_stage4.sh
./scripts/run_lss_post_pass_campaign_stage5.sh
./scripts/run_lss_post_pass_campaign_stage6.sh
./scripts/run_lss_post_pass_campaign_finalize.sh
./scripts/run_lss_canonization_campaign_register.sh
./scripts/run_lss_canonization_campaign_stage2.sh
./scripts/run_lss_canonization_campaign_stage3.sh
./scripts/run_lss_canonization_campaign_stage4.sh
./scripts/run_lss_canonization_campaign_stage5.sh
./scripts/run_lss_canonization_campaign_stage6.sh
./scripts/run_lss_canonization_campaign_finalize.sh
./scripts/run_lss_hard_family_register.sh
./scripts/run_lss_hard_family_stage2.sh
./scripts/run_lss_hard_family_stage3.sh
./scripts/run_lss_hard_family_stage4.sh
./scripts/run_lss_hard_family_stage5.sh
./scripts/run_lss_hard_family_stage6.sh
./scripts/run_lss_hard_family_stage7.sh
./scripts/run_lss_hard_family_stage8.sh
./scripts/run_lss_hard_family_finalize.sh
./scripts/run_lss_hard_family_saturation_register.sh
./scripts/run_lss_hard_family_saturation_stage1.sh
./scripts/run_lss_hard_family_saturation_stage2.sh
./scripts/run_lss_hard_family_saturation_stage3.sh
./scripts/run_lss_hard_family_saturation_stage4.sh
./scripts/run_lss_hard_family_saturation_stage5.sh
./scripts/run_lss_hard_family_saturation_stage6.sh
./scripts/run_lss_hard_family_saturation_stage7.sh
./scripts/run_lss_hard_family_saturation_finalize.sh
./scripts/run_lss_successor_migration_register.sh
./scripts/run_lss_successor_migration_stage1.sh
./scripts/run_lss_successor_migration_stage2.sh
./scripts/run_lss_successor_migration_stage3.sh
./scripts/run_lss_successor_migration_stage4.sh
./scripts/run_lss_successor_migration_stage5.sh
./scripts/run_lss_successor_migration_stage6.sh
./scripts/run_lss_successor_migration_stage7.sh
./scripts/run_lss_successor_migration_finalize.sh
```

Resume a run from a checkpoint:

```bash
python -m psmn_rl.train \
  --config configs/experiments/minigrid_doorkey_flat_dense.yaml \
  --resume-from outputs/experiments/baselines/minigrid_doorkey_flat_dense/latest.pt
```

Compare greedy and sampled evaluation for a checkpoint:

```bash
./scripts/eval_policy_modes.sh \
  outputs/experiments/baselines/minigrid_doorkey_token_dense/latest.pt \
  configs/experiments/minigrid_doorkey_token_dense.yaml \
  16
```

Current high-level findings are summarized in `summary.md` and `report.md` at the repo root, with detailed phase artifacts under `outputs/reports/`.

Current accepted state:

- active benchmark: `round6` on DoorKey only
- public claim envelope: still teacher-guided, KL learner-state, DoorKey only, external `64`-episode evaluation only
- internal adjacent-task candidate: `partial225_greedy` on `Memory`, with task-scoped benchmark-candidate status only

See `outputs/reports/memory_next_decision_memo.md`, `outputs/reports/memory_next_candidate_pack.json`, and `outputs/reports/memory_next_candidate_canonicalization.md` for the current Memory-side status.

## Frozen Benchmark Pack

The DoorKey claim is now sealed as a frozen benchmark pack rather than a loose collection of reports.

- human-readable pack: `outputs/reports/frozen_benchmark_pack.md`
- machine-readable pack: `outputs/reports/frozen_benchmark_pack.json`
- schema report: `outputs/reports/benchmark_pack_schema_report.md`
- candidate pack schema + template:
  - `outputs/reports/candidate_result_pack_schema.md`
  - `outputs/reports/candidate_result_pack_template.json`

Validate the sealed pack with one command:

```bash
./scripts/run_frozen_benchmark_pack_validation.sh
```

## Active DoorKey Benchmark

The active canonical DoorKey benchmark is now the `round6` teacher-guided KL learner-state successor, while the frozen benchmark pack remains archived as the legacy baseline.

- current confirmation decision: `outputs/reports/portfolio_decision_memo.md`
- active benchmark pack: `outputs/reports/portfolio_candidate_pack.json`
- current gate report: `outputs/reports/portfolio_gate_report.md`
- archived frozen pack: `outputs/reports/frozen_benchmark_pack.json`

### Frozen Frontier Operations

The current measured DoorKey restart frontier is also frozen as an operational bundle.

- frontier contract: `outputs/reports/portfolio_frontier_contract.json`
- frontier guard report: `outputs/reports/portfolio_frontier_guard_report.md`
- frontier guard workflow contract: `outputs/reports/portfolio_frontier_guard_workflow_contract.md`
- frontier active-state doctor: `outputs/reports/portfolio_active_state_doctor.md`
- frontier operational state: `outputs/reports/portfolio_operational_state.md`
- frontier schedule: `outputs/reports/portfolio_frontier_schedule.md`
- frontier kit: `outputs/reports/portfolio_frontier_kit.json`
- frontier seed pack: `outputs/reports/portfolio_seed_pack.json`
- frontier seed-pack doctor: `outputs/reports/portfolio_seed_pack_doctor.md`
- default bounded restart prior: `round7`
- replay-validated alternate: `round10`

Run the local frontier guard:

```bash
bash ./scripts/run_portfolio_frontier_guard.sh
```

Render the frontier guard workflow contract directly:

```bash
python -m psmn_rl.analysis.portfolio_frontier_guard_workflow_contract \
  --output outputs/reports/portfolio_frontier_guard_workflow_contract.md \
  --json outputs/reports/portfolio_frontier_guard_workflow_contract.json \
  --fail-on-drift
```

Run the doctor directly:

```bash
python -m psmn_rl.analysis.portfolio_frontier_doctor \
  --output outputs/reports/portfolio_frontier_doctor.md \
  --json outputs/reports/portfolio_frontier_doctor.json \
  --fail-on-drift
```

Run the seed-pack doctor directly:

```bash
python -m psmn_rl.analysis.portfolio_seed_pack_doctor \
  --output outputs/reports/portfolio_seed_pack_doctor.md \
  --json outputs/reports/portfolio_seed_pack_doctor.json \
  --fail-on-drift
```

Run the active-state doctor directly:

```bash
python -m psmn_rl.analysis.portfolio_active_state_doctor \
  --output outputs/reports/portfolio_active_state_doctor.md \
  --json outputs/reports/portfolio_active_state_doctor.json \
  --fail-on-drift
```

Render the current operational snapshot directly:

```bash
python -m psmn_rl.analysis.portfolio_operational_state \
  --output outputs/reports/portfolio_operational_state.md \
  --json outputs/reports/portfolio_operational_state.json
```

The consolidated guard status is written to:

- `outputs/reports/portfolio_frontier_guard_report.md`
- `outputs/reports/portfolio_frontier_guard_report.json`
- `outputs/reports/portfolio_frontier_guard_workflow_contract.md`
- `outputs/reports/portfolio_frontier_guard_workflow_contract.json`
- `outputs/reports/portfolio_active_state_doctor.md`
- `outputs/reports/portfolio_active_state_doctor.json`
- `outputs/reports/portfolio_operational_state.md`
- `outputs/reports/portfolio_operational_state.json`
- `outputs/reports/portfolio_seed_pack_doctor.md`
- `outputs/reports/portfolio_seed_pack_doctor.json`

Run the pack-based gate against a candidate pack:

```bash
./scripts/run_claim_gate.sh \
  outputs/reports/frozen_benchmark_pack.json \
  outputs/reports/frozen_candidate_result_pack.json \
  outputs/reports/claim_gate_pack_dry_run.md \
  outputs/reports/claim_gate_pack_dry_run.json
```

Ad hoc markdown summaries are not enough for thaw consideration. Future DoorKey candidates must be packaged with the candidate-pack schema, run through the pack-based gate, and clear the adversarial conformance lane:

```bash
bash ./scripts/run_claim_gate_conformance.sh
```

Historical trust now also includes replaying the gate against the repo's real claim history and the later saturation-scale hard-family canonization result:

- earlier hard-family boundary test:
  - [hard_family_definition.md](outputs/reports/hard_family_definition.md)
  - [hard_family_campaign_registration.md](outputs/reports/hard_family_campaign_registration.md)
  - [hard_family_stage2_dev_screening.md](outputs/reports/hard_family_stage2_dev_screening.md)
  - [hard_family_stage3_fairness.md](outputs/reports/hard_family_stage3_fairness.md)
  - [hard_family_canonization_decision_memo.md](outputs/reports/hard_family_canonization_decision_memo.md)
- current saturation-scale canonization result:
  - [hard_family_saturation_definition.md](outputs/reports/hard_family_saturation_definition.md)
  - [hard_family_saturation_stage1_screening.md](outputs/reports/hard_family_saturation_stage1_screening.md)
  - [hard_family_saturation_stage3_holdout.md](outputs/reports/hard_family_saturation_stage3_holdout.md)
  - [hard_family_saturation_gate_report.md](outputs/reports/hard_family_saturation_gate_report.md)
  - [hard_family_saturation_decision_memo.md](outputs/reports/hard_family_saturation_decision_memo.md)

```bash
bash ./scripts/run_claim_history_replay.sh
```

That replay is the check that the current sealed gate still agrees with the accepted historical claim trajectory rather than only with synthetic red-team cases.

## Current DoorKey Result

The current repo conclusion is still control-first, and the frozen benchmark pack remains the baseline comparison unit for pack-based comparisons. Inside that baseline, the saturation-scale hard-family program produced a successor candidate that is now qualified for canonization within DoorKey only.

- `flat_dense` remains the best verified greedy DoorKey control.
- `token_dense` with `ppo.ent_coef=0.001` remains the canonical recovered tokenized control.
- PPO-only `SARE` is still greedy-negative.
- `post_unlock_weighted` remains the first thaw-qualified DoorKey candidate:
  - it keeps a gate `PASS` against the frozen benchmark pack
  - it survives expanded fairness, route validation, and longitudinal stability
  - it created the within-envelope successor line, but it did not itself become canonical because `post_pass_b` stayed token-dense-led
- The saturation-scale hard-family program then promoted the `round6` successor inside that same family:
  - hard-family dev KL learner-state `SARE`: `1.0000` versus matched `token_dense` `1.0000`
  - hard-family holdout KL learner-state `SARE`: `1.0000` versus matched `token_dense` `1.0000`
  - frozen-comparable combined KL learner-state `SARE`: `1.0000`
  - route validation: `pass`
  - stability: `stable_plateau` on dev, holdout, and healthy probes
  - decision: `qualified for canonization within DoorKey`
- Offline teacher distillation and other bounded recovery families remain negative.
- The routed result that survives is still teacher-guided and bounded:
  - teacher-logit KL learner-state supervision for `SARE`
  - DoorKey only
  - external `64`-episode evaluation only
- The current canonization-qualified successor pack is:
  - candidate: `round6`
  - retry-block KL learner-state `SARE` mean on `47/53/59`: `1.0000`
  - same-block KL learner-state `single_expert` mean: `1.0000`
  - frozen-comparable combined DoorKey KL learner-state `SARE` mean: `1.0000`
  - pack verdict: `PASS: thaw consideration allowed`
- The larger benchmark-expansion program then confirmed `round6` as the active benchmark under a broader DoorKey family:
  - thirty challenger runs across fifteen bounded mechanism directions advanced only four verified ties, and none survived matched fairness
  - holdout DoorKey KL learner-state `SARE`: `0.9154` versus matched `token_dense` `0.9167` and matched `single_expert` `0.9167`
  - healthy-family KL learner-state `SARE`: `1.0000` versus matched `token_dense` `0.8854` and matched `single_expert` `0.8889`
  - route validation: `pass`
  - stability: `stable_plateau`, with one holdout probe that is a flat-zero plateau rather than a late-round spike
  - exploratory KeyCorridor track: `clearly negative`
  - decision: `round6 confirmed as active DoorKey benchmark and internal DoorKey envelope strengthened`
- The later 50/50 portfolio campaign then re-confirmed the same benchmark under a broader fruitful/exploratory split:
  - forty substantive challenger runs split `20/20` fruitful vs exploratory advanced only verified `0.8889` ties
  - no challenger survived matched fairness
  - holdout, healthy anti-regression, route validation, and stability all stayed favorable for `round6`
  - exploratory KeyCorridor remained `clearly negative`
  - decision: `round6 confirmed as active DoorKey benchmark and internal DoorKey benchmark state strengthened`
- The later 80-run next-mega saturation campaign then forced the narrower current reading:
  - eighty substantive challenger runs split `40/40` fruitful vs exploratory advanced no Stage 1 challenger
  - holdout did not strengthen `round6` over matched controls
  - healthy anti-regression, route validation, and stability still passed
  - exploratory KeyCorridor remained `clearly negative`
  - decision: `benchmark/frontier state needs narrowing`
- The accepted live repo state is now repaired around that narrower outcome:
  - current decision reference: `outputs/reports/next_mega_portfolio_decision_memo.md`
  - current gate report: `outputs/reports/portfolio_gate_report.md`
  - current gate-repair memo: `outputs/reports/next_round_gate_repair.md`
  - current gate reference pack: `outputs/reports/round6_current_benchmark_pack.json`
- That result is still an extraction-method result, not a PPO-alone routed win.
- The allowed current interpretation is still narrow:
  - active canonical DoorKey benchmark within the existing teacher-guided KL learner-state envelope
  - no PPO-only routed claim
  - no specifically multi-expert routed claim
  - no cross-task claim
  - no KeyCorridor transfer claim

Canonical reports for the current phase:

- `outputs/reports/frozen_claim_envelope.md`
- `outputs/reports/frozen_claim_manifest_report.md`
- `outputs/reports/frozen_baseline_validation.md`
- `outputs/reports/claim_gate_dry_run.md`
- `outputs/reports/claim_ledger.md`
- `outputs/reports/next_round_state_reconciliation.md`
- `outputs/reports/next_round_baseline_sync.md`
- `outputs/reports/next_round_gate_repair.md`
- `outputs/reports/round6_current_benchmark_pack.md`
- `outputs/reports/round6_current_benchmark_pack.json`
- `outputs/reports/round6_current_benchmark_pack_validation.md`
- `outputs/reports/future_retry_template.md`
- `outputs/reports/freeze_hardening_decision_memo.md`
- `outputs/reports/claim_gate_conformance_report.md`
- `outputs/reports/claim_gate_redteam_decision_memo.md`
- `outputs/reports/lss_forensic_atlas_reproduction_note.md`
- `outputs/reports/long_campaign_registration.md`
- `outputs/reports/long_campaign_stage2_screening.md`
- `outputs/reports/long_campaign_stage3_fairness.md`
- `outputs/reports/long_campaign_stage4_replication.md`
- `outputs/reports/long_campaign_stage5_route_validation.md`
- `outputs/reports/long_campaign_candidate_pack.json`
- `outputs/reports/long_campaign_gate_report.md`
- `outputs/reports/long_campaign_decision_memo.md`
- `outputs/reports/canonization_campaign_registration.md`
- `outputs/reports/canonization_stage2_hard_block_screening.md`
- `outputs/reports/canonization_stage3_hard_block_fairness.md`
- `outputs/reports/canonization_gate_report.md`
- `outputs/reports/canonization_decision_memo.md`
- `outputs/reports/hard_family_saturation_definition.md`
- `outputs/reports/hard_family_saturation_registration.md`
- `outputs/reports/hard_family_saturation_stage1_screening.md`
- `outputs/reports/hard_family_saturation_stage2_fairness.md`
- `outputs/reports/hard_family_saturation_stage3_holdout.md`
- `outputs/reports/hard_family_saturation_stage4_antiregression.md`
- `outputs/reports/hard_family_saturation_stage5_route_validation.md`
- `outputs/reports/hard_family_saturation_stage6_stability.md`
- `outputs/reports/hard_family_saturation_successor_pack.json`
- `outputs/reports/hard_family_saturation_gate_report.md`
- `outputs/reports/hard_family_saturation_decision_memo.md`
- `outputs/reports/portfolio_stage1_screening_fruitful.md`
- `outputs/reports/portfolio_stage1_screening_exploratory.md`
- `outputs/reports/portfolio_stage3_fairness.md`
- `outputs/reports/portfolio_stage4_holdout.md`
- `outputs/reports/portfolio_stage5_antiregression.md`
- `outputs/reports/portfolio_stage6_route_validation.md`
- `outputs/reports/portfolio_stage7_stability.md`
- `outputs/reports/portfolio_stage8_exploratory_transfer.md`
- `outputs/reports/portfolio_candidate_pack.json`
- `outputs/reports/portfolio_gate_report.md`
- `outputs/reports/portfolio_decision_memo.md`
- `outputs/reports/portfolio_frontier_contract.json`
- `outputs/reports/lss_forensic_casebook.md`
- `outputs/reports/lss_forensic_round_audit.md`
- `outputs/reports/lss_forensic_route_locality.md`
- `outputs/reports/lss_resume_qualification_scorecard.md`
- `outputs/reports/lss_forensic_atlas_decision_memo.md`
- `outputs/reports/lss_final_block_single_expert_control_report.md`
- `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.md`

## Frozen Claim Scope

Allowed:

- bounded teacher-guided DoorKey `SARE` result only

Not allowed:

- PPO-only routed win
- specifically multi-expert routed advantage
- cross-task routed advantage
- KeyCorridor transfer claim

Future thaw candidates must:

- use the external `64`-episode `policy_diagnostics` path
- beat the frozen retry-block KL learner-state `SARE` mean `0.3125` on seeds `47/53/59`
- at least match matched KL learner-state `single_expert` on that same block
- preserve the combined DoorKey KL learner-state `SARE` mean `0.7122`

## Claim Gate

- Frozen pack validation verdict: `PASS: frozen benchmark pack validated`
- Current pack-based gate verdict on the frozen candidate: `FAIL: claim remains frozen`
- Malformed candidate packs are rejected as `INCONCLUSIVE: missing prerequisites`

Canonical operational artifacts:

- `outputs/reports/frozen_benchmark_pack_validation.md`
- `outputs/reports/claim_gate_pack_dry_run.md`
- `outputs/reports/claim_gate_pack_inconclusive.md`
- `outputs/reports/freeze_hardening_operational_memo.md`

Canonical gate artifacts:

- `configs/claims/doorkey_frozen_claim.yaml`
- `outputs/reports/frozen_claim_envelope.md`
- `outputs/reports/frozen_claim_manifest_report.md`
- `outputs/reports/frozen_baseline_validation.md`
- `outputs/reports/claim_gate_dry_run.md`

## Repository Layout

Core code lives in `src/psmn_rl/`.

- `envs/`: environment factories and wrappers
- `models/`: encoders, cores, routing, relational, options, and heads
- `rl/`: PPO, rollout storage, and DDP utilities
- `analysis/`: run summarization utilities

Experiment configs live in `configs/`. Output artifacts are written under `outputs/`.

## Reproducibility

- Save resolved configs with every run.
- Rank 0 alone writes checkpoints and summaries.
- Route statistics are logged together with reward and throughput.
- Use Beads for milestones and experiment tracking.
