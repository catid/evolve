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
