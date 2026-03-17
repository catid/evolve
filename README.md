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

## Current DoorKey Result

The current repo conclusion is still control-first, but the routed story has now split cleanly:

- `flat_dense` remains the best verified greedy DoorKey control.
- `token_dense` with `ppo.ent_coef=0.001` remains the canonical recovered tokenized control.
- PPO-only `SARE` is still greedy-negative.
- Offline teacher distillation and other bounded recovery families remain negative.
- The only positive routed result is still narrower and teacher-guided:
  - teacher-logit KL learner-state supervision for `SARE`
  - DoorKey only
  - external `64`-episode evaluation only
- After the forensic-atlas pass, the best reading of that result is:
  - on the final fresh block `47/53/59`, matched KL learner-state `single_expert` reaches mean greedy success `0.4635` versus `0.3125` for KL learner-state `SARE`
  - the deep forensic package shows a mixed failure signature rather than one clean retry lever: seed `47` is the clearest route-fragile `SARE` case, while `53/59` look closer to shared structured-student post-unlock collapse
  - the full combined DoorKey picture still leaves KL learner-state `SARE` slightly ahead:
    - `KL` learner-state `token_dense`: mean greedy success `0.6354`
    - `KL` learner-state `single_expert`: mean greedy success `0.6862`
    - `KL` learner-state `SARE`: mean greedy success `0.7122`
  - that edge is still too small and too final-block-sensitive to justify either a bounded resume attempt or a specifically multi-expert routed DoorKey claim
- That positive result is still an extraction-method result, not a PPO-alone routed win.
- The current repo recommendation is to stay frozen at this scope rather than reopen the DoorKey claim.

Canonical reports for the current phase:

- `outputs/reports/lss_forensic_atlas_reproduction_note.md`
- `outputs/reports/lss_forensic_casebook.md`
- `outputs/reports/lss_forensic_round_audit.md`
- `outputs/reports/lss_forensic_route_locality.md`
- `outputs/reports/lss_resume_qualification_scorecard.md`
- `outputs/reports/lss_forensic_atlas_decision_memo.md`
- `outputs/reports/lss_final_block_single_expert_control_report.md`
- `outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.md`

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
