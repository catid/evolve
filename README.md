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
- The only reopened routed result is narrower and teacher-guided:
  - teacher-logit KL learner-state supervision for `SARE`
  - external 64-episode DoorKey gate
  - original `7/11/19` mean greedy success `0.7135` for routed `SARE` vs `0.6667` for matched KL learner-state `token_dense`
  - combined original+fresh six-seed DoorKey mean greedy success `0.8568` for routed `SARE` vs `0.6042` for matched KL learner-state `token_dense`
  - no routed DoorKey seed remains at `0.0`
- That reopened claim is an extraction-method result, not a PPO-alone routed win.
- The current claim-consolidation phase strengthens that DoorKey result with two extra checks:
  - fresh matched teacher-guided `token_dense` controls on DoorKey seeds `23/29/31`
  - causal route-dependence probes on recovered `SARE` checkpoints

Current bounded-positive DoorKey result:

- teacher-guided `KL` learner-state `SARE` still holds the strongest routed DoorKey result in the repo
- the missing matched `single_expert` control does not erase that DoorKey edge on the original lane:
  - original seeds `7/11/19`: `KL` learner-state `single_expert` mean greedy success `0.6667`
  - original seeds `7/11/19`: `KL` learner-state `SARE` mean greedy success `0.7135`
- one more fresh matched seed block keeps the edge alive:
  - fresh-extra seeds `37/41/43`: `KL` learner-state `token_dense` mean greedy success `0.3333`
  - fresh-extra seeds `37/41/43`: `KL` learner-state `SARE` mean greedy success `0.8229`
- the expanded nine-seed DoorKey picture still favors routed `SARE`:
  - `KL` learner-state `token_dense`: mean greedy success `0.5139`
  - `KL` learner-state `SARE`: mean greedy success `0.8455`
  - no routed DoorKey seed remains at `0.0`
- causal routing dependence now extends beyond the original `7/23` demonstration:
  - expert ablation and fixed-router override still collapse or severely damage recovered success on seeds `7`, `19`, `23`, and `29`
  - route randomization remains catastrophic on `7`, `19`, and `23`, but is only weakly harmful on fresh seed `29`
- the same method still shows no bounded KeyCorridor transfer, so the claim remains DoorKey-specific

Canonical reports for the current phase:

- `outputs/reports/lss_claim_broadening_reproduction_note.md`
- `outputs/reports/lss_single_expert_matched_control_report.md`
- `outputs/reports/lss_extended_route_dependence_report.md`
- `outputs/reports/lss_additional_fresh_seed_block_report.md`
- `outputs/reports/lss_expanded_combined_doorkey_report.md`
- `outputs/reports/lss_claim_broadening_decision_memo.md`

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
