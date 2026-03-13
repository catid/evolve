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

Current high-level findings are summarized in `summary.md`, `report.md`, and `overfit_report.md` at the repo root.

## Current DoorKey Result

The current repo conclusion is control-first and negative on routed claims:

- `flat_dense` remains the best verified greedy DoorKey control.
- The original tokenized DoorKey gap was split between a weak `token_dense` control and softer greedy extraction in `single_expert` / `SARE`.
- The smallest successful recovery was `token_dense` with `ppo.ent_coef=0.001`.
- Under that recovered setting, `token_dense` reaches greedy DoorKey success `0.75`.
- A fair matched `SARE` rerun on the same setting still has greedy DoorKey success `0.0`.

Canonical reports for this phase:

- `outputs/reports/reproduction_note.md`
- `outputs/reports/policy_extraction_report.md`
- `outputs/reports/tokenization_gap_report.md`

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
