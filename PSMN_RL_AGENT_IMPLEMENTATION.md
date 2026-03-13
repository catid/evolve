# PSMN RL Implementation Spec and Agent Prompt

## Purpose

Build a research codebase that tests **Packet-Switched Morphogenic Network (PSMN)** ideas as **reinforcement learning agents for games**, not as language models.

The initial scope is to implement **small, falsifiable experiments** that compare a dense tokenized baseline against four routed/modular variants:

1. **SARE** — Static Asymmetric Routed Experts (one-hop routed micro-experts)
2. **TREG-H** — Two-Hop Routed Expert Graph with halting / adaptive depth
3. **SRW** — Selective Relational Widgets (mostly unary experts plus a small routed relational module)
4. **POR** — Persistent Option Routing across environment steps

The project must be engineered for:

- single-node **multi-GPU** training using **all visible GPUs**
- reproducible RL experiments
- clear ablations and metrics
- task tracking with **Beads** (`bd init`)
- durable project guidance in `AGENTS.md`

This document is intended to be directly usable by an implementation agent.

---

## Non-negotiable constraints

- Use **Python 3.12**.
- Use **`uv venv`** for the virtual environment.
- Use the **latest PyTorch nightly for CUDA 13.0 (`cu130`)**.
- Use **Beads** for task tracking, initialized with **`bd init`** at repo root.
- Use **all visible GPUs** by default with `torchrun` + `DistributedDataParallel`.
- Keep `AGENTS.md` updated with durable repo instructions and experiment conventions.
- Start with **small RL experiments** on MiniGrid, then optionally Procgen.
- Prefer **simple, inspectable implementations** over frameworks that hide control flow.
- Optimize for **fast falsification** before novelty.

---

## Research objective

We want to test whether conditional, routed, token-based computation helps RL agents on small game tasks, relative to tokenized dense baselines, under matched or compute-normalized budgets.

The project should answer these questions:

1. Does one-hop routing beat a tokenized dense control?
2. Does sequential expert composition with halting help on compositional tasks?
3. Does selectively invoking relational computation help when object interactions matter?
4. Does temporally persistent routing / option-like structure help on partially observable or sparse-reward tasks?
5. Are the learned paths stable, sparse, and interpretable enough to justify later work on morphogenesis and compression?

---

## Deliverables

The agent should produce the following:

1. A working Python package under `src/psmn_rl/`.
2. A reproducible environment setup using `uv venv` and Python 3.12.
3. Multi-GPU training entrypoints using `torchrun` and DDP.
4. Implementations of:
   - tokenized dense baseline
   - tokenized single-expert baseline
   - SARE
   - TREG-H
   - SRW
   - POR
5. PPO training loop first; code structured so IMPALA can be added later.
6. MiniGrid experiment suite first; Procgen integration second.
7. Logging of training curves, wall-clock, GPU utilization, route statistics, path coverage, expert usage, and compute-normalized metrics.
8. Config files for all experiments.
9. Unit tests and smoke tests.
10. A concise README for running experiments.
11. A maintained `AGENTS.md`.
12. A Beads task graph seeded with epics and child tasks.

---

## Suggested repository layout

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .python-version
├── .gitignore
├── configs/
│   ├── baseline/
│   ├── sare/
│   ├── treg_h/
│   ├── srw/
│   └── por/
├── scripts/
│   ├── bootstrap.sh
│   ├── train_minigrid.sh
│   ├── train_procgen.sh
│   ├── eval.sh
│   └── seed_beads.sh
├── src/psmn_rl/
│   ├── __init__.py
│   ├── envs/
│   ├── data/
│   ├── models/
│   │   ├── encoders/
│   │   ├── cores/
│   │   ├── routing/
│   │   ├── relational/
│   │   ├── options/
│   │   └── heads/
│   ├── rl/
│   │   ├── ppo/
│   │   ├── rollout/
│   │   ├── losses/
│   │   └── distributed/
│   ├── train.py
│   ├── evaluate.py
│   ├── launch.py
│   ├── metrics.py
│   ├── logging.py
│   ├── utils/
│   └── analysis/
├── tests/
│   ├── test_configs.py
│   ├── test_models.py
│   ├── test_routing.py
│   ├── test_ddp_smoke.py
│   └── test_minigrid_smoke.py
└── outputs/
```

---

## Environment bootstrap

### System assumptions

- Linux
- NVIDIA GPUs available
- CUDA drivers already installed by the user / server image
- `uv` available or installable
- `bd` available or installable system-wide

### Bootstrap commands

Use these commands exactly unless there is a compatibility issue.

```bash
# 1) Python 3.12
uv python install 3.12

# 2) Virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# 3) Core packaging tools
uv pip install -U pip setuptools wheel

# 4) PyTorch nightly cu130
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130

# 5) Project deps
uv pip install -e .
```

If the implementation needs additional packages for image ops or benchmarking, add them in `pyproject.toml` and reinstall with `uv pip install -e .`.

### Beads initialization

If `bd` is not installed yet, install it system-wide first. Then initialize task tracking at repo root:

```bash
bd init
```

If prompted, choose the repo-maintainer / primary-owner mode appropriate for a local research repo, and accept recommended hooks unless there is a concrete reason not to.

---

## Beads workflow

The agent must use Beads for all substantial work.

### Required policy

- Every milestone must have a Beads issue.
- Every substantial subtask must have a Beads issue.
- Use dependencies to express blockers and parent-child structure.
- Claim a task before working on it.
- Add comments with progress and blockers.
- Close tasks with a summary of what changed and what evidence was gathered.

### Seed tasks

Create these tasks after `bd init`:

```bash
bd create "Bootstrap repo, environment, and Beads workflow" -p 0 -t task
bd create "Implement tokenized dense RL baseline" -p 0 -t task
bd create "Implement tokenized single-expert baseline" -p 1 -t task
bd create "Implement SARE routed core" -p 0 -t task
bd create "Implement TREG-H routed core" -p 1 -t task
bd create "Implement SRW routed relational core" -p 1 -t task
bd create "Implement POR option-routing core" -p 1 -t task
bd create "Implement PPO + DDP training stack" -p 0 -t task
bd create "Integrate MiniGrid experiments" -p 0 -t task
bd create "Integrate Procgen experiments" -p 2 -t task
bd create "Add metrics, logging, and analysis tooling" -p 0 -t task
bd create "Write smoke tests and reproducibility checks" -p 0 -t task
bd create "Run baseline and routed ablations" -p 0 -t task
bd create "Write README and experiment guide" -p 1 -t task
```

Then connect dependencies using `bd dep add <child> <parent>`.

### Minimum Beads commands the agent should use

```bash
bd ready
bd list
bd show <id>
bd update <id> --claim
bd comments add <id> "progress update"
bd dep add <child> <parent>
bd close <id> --reason "done"
```

---

## AGENTS.md requirements

Create or update `AGENTS.md` immediately. It should include:

- project purpose and active research questions
- hard constraints (Python 3.12, uv venv, PyTorch nightly cu130, Beads, DDP, use all GPUs)
- the four initial experimental variants
- coding conventions
- experiment conventions
- reproducibility rules
- route / expert metrics to log
- what to do before opening a PR / finalizing a result
- instructions to prefer small falsifiable experiments before expanding scope

An initial `AGENTS.md` template is provided separately and should be used.

---

## Training and distributed execution

### Use all GPUs by default

The implementation must auto-detect all visible GPUs and use them.

Preferred launch style:

```bash
NPROC_PER_NODE=$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m psmn_rl.launch \
  --config configs/baseline/minigrid_dense.yaml
```

### DDP requirements

- Use `torch.distributed` + `DistributedDataParallel`.
- One process per GPU.
- Correct seeding per rank.
- Use distributed-aware logging and checkpointing.
- Save checkpoints only from rank 0.
- Barrier where needed.
- Make rollout collection and PPO minibatch sharding deterministic enough for reproducibility.

### Precision and performance

- Use `bf16` autocast if supported.
- Fall back to `fp16` or `fp32` safely.
- Enable `torch.set_float32_matmul_precision("high")`.
- Use pinned memory and non-blocking transfers where helpful.
- Keep the initial code simple; do not prematurely optimize with custom kernels.

---

## Baselines and experiment design

### Mandatory baselines

For every routed variant, compare against:

1. flattened dense baseline
2. tokenized dense baseline
3. tokenized single-expert baseline

Do not skip the tokenized dense control.

### Environments

#### Phase 1: MiniGrid

Start here:

- DoorKey
- KeyCorridor
- Memory
- DynamicObstacles

#### Phase 2: Procgen

Only after MiniGrid works:

- CoinRun
- Heist
- Maze

### RL algorithm

- Implement PPO first.
- Keep interfaces clean enough to add IMPALA later.

### Shared observation format

Observations should be encoded into **tokens** where possible.

Support at least:

- grid / object tokens for MiniGrid
- image patch or compact spatial tokens for visual tasks

---

## Model variants to implement

## 1) Tokenized dense baseline

A straightforward tokenized encoder + dense core + actor/critic heads.

Purpose:

- establish whether tokenization alone explains most gains
- provide a clean control for all routed variants

## 2) Tokenized single-expert baseline

Same overall routing/dataflow shape as routed models, but with only one shared expert.

Purpose:

- isolate the benefit of routing vs architectural overhead

## 3) SARE

### Core idea

One-hop routing over micro-experts using asymmetric query-to-key matching.

### Approximate math

For token `z_i`:

```text
q_i = W_q z_i
s_{ie} = q_i^T k_e / sqrt(d_a)
p_{ie} = softmax(s_{ie} / tau)
y_i = sum_{e in TopK(i)} p_{ie} f_e(z_i)
```

### Requirements

- soft or soft-top-k routing first
- expert usage stats
- route entropy
- path coverage of top-K routes

## 4) TREG-H

### Core idea

Allow one or two expert hops with a halting head after the first hop.

### Requirements

- learned transition bias between experts
- halting probability head
- ponder penalty / compute penalty
- report average hop count and hop count by task state bucket

## 5) SRW

### Core idea

Mostly unary routed experts, but a small relational module is invoked only when a gating signal says cross-token interaction is needed.

### Requirements

- route / gate into relational widget only for selected states
- select a subset of salient tokens for the relational module
- simplest relational module can be a tiny self-attention block over selected tokens
- log relational invocation rate

## 6) POR

### Core idea

Persist routing decisions / route templates across multiple environment steps, like options.

### Requirements

- option policy
- option termination head
- actor/critic conditioned on option
- track option duration and switch rate

---

## Metrics and logging

For every run, log:

- episode return
- success rate where applicable
- environment steps
- wall-clock
- throughput
- GPU utilization (if easy to collect)
- active parameters / approximate compute per forward
- expert load histogram
- route entropy
- path entropy
- coverage of top 10 / 20 / 50 routes
- average hop count
- average halting probability
- relational widget usage rate
- option duration and option switch rate
- seed variance

Produce analysis scripts that can answer:

- which variant wins by environment?
- are gains still present when normalized by compute?
- do route patterns correlate with task phase?
- are learned paths stable across seeds?

---

## Acceptance criteria

The first milestone is complete when all of the following are true:

1. `uv venv` setup works on a fresh machine with Python 3.12.
2. PyTorch nightly cu130 is installed and GPUs are detected.
3. `bd init` has been run and seeded tasks exist.
4. `AGENTS.md` exists and is useful.
5. DDP launches across all visible GPUs.
6. Tokenized dense baseline trains on one MiniGrid task.
7. SARE trains on the same task.
8. Metrics and route statistics are logged.
9. A smoke-test command completes in a short time on 1 GPU or CPU.
10. README explains how to reproduce the first experiment.

The second milestone is complete when:

- TREG-H, SRW, and POR all run on MiniGrid
- at least one routed variant is benchmarked against dense/tokenized controls on 2–4 environments
- results are saved and summarized in markdown

---

## Implementation strategy

### Order of work

1. Bootstrap repo, env, packaging, Beads, AGENTS.md
2. Build PPO + DDP skeleton
3. Add MiniGrid integration and tokenized dense baseline
4. Add tokenized single-expert baseline
5. Add SARE
6. Add TREG-H
7. Add SRW
8. Add POR
9. Add Procgen
10. Add analysis scripts and experiment summary docs

### Coding style

- Prefer explicit, readable PyTorch.
- Keep routing logic isolated in `models/routing/`.
- Keep RL algorithm code isolated from model code.
- Add shape assertions in critical places.
- Add docstrings for nontrivial classes.
- Prefer typed dataclasses or simple config objects.
- Avoid over-engineered framework abstractions.

### Testing

At minimum:

- config parsing test
- routed forward pass shape tests
- deterministic seed smoke test
- one-batch PPO update smoke test
- DDP smoke test
- MiniGrid training smoke test

---

## What not to implement yet

Do **not** start with these:

- morphogenic split / merge / prune / compress
- delayed message buffers across time
- full attention-as-widget unification
- custom Triton kernels
- gigantic benchmark suites

These are second-wave features and will confound attribution.

---

## Prompt to give an implementation agent

Copy the following prompt into the coding agent.

```text
You are implementing a research codebase for Packet-Switched Morphogenic Network (PSMN) experiments in reinforcement learning games.

Your job is to build a clean, reproducible Python project that tests whether routed token-based computation beats tokenized dense baselines on small RL tasks.

Hard constraints:
- Python 3.12 only.
- Use uv venv, not conda.
- Use latest PyTorch nightly cu130.
- Use all visible GPUs with torchrun + DistributedDataParallel.
- Initialize Beads with bd init and use it for task tracking.
- Create and maintain AGENTS.md with durable instructions and project memory.
- Start with MiniGrid, then Procgen.
- Implement PPO first.
- Prefer small falsifiable experiments before expanding scope.

You must build these variants:
1. tokenized dense baseline
2. tokenized single-expert baseline
3. SARE
4. TREG-H
5. SRW
6. POR

Required workflow:
1. Bootstrap the repo and environment.
2. Run bd init at repo root.
3. Seed Beads tasks for each milestone and subtask.
4. Create or update AGENTS.md immediately.
5. Implement PPO + DDP skeleton.
6. Implement MiniGrid tokenized dense baseline first.
7. Add the routed variants one by one.
8. Add logging and analysis of route statistics.
9. Write smoke tests.
10. Keep README and AGENTS.md current.

Required engineering standards:
- readable PyTorch code
- typed configs where practical
- route logic isolated from RL loop
- deterministic seeds where possible
- DDP-safe checkpointing and logging
- no direct database access for Beads; use bd CLI
- no custom kernels in the first pass

Required experiment standards:
- compare every routed model to flattened dense, tokenized dense, and tokenized single-expert controls
- log return, wall-clock, compute-normalized performance, expert load, path entropy, path coverage, hop count, relational usage, and option durations
- save configs and summaries for every run

Deliverables:
- working package under src/psmn_rl/
- pyproject.toml
- scripts for training and evaluation
- configs for all variants
- tests
- README.md
- AGENTS.md
- Beads task graph

Execution order:
1. bootstrap environment
2. MiniGrid + PPO + tokenized dense baseline
3. tokenized single-expert baseline
4. SARE
5. TREG-H
6. SRW
7. POR
8. Procgen integration
9. experiment summaries

Be conservative and falsification-oriented. If a simpler baseline answers the question, prefer that. Do not expand into morphogenesis or custom kernels unless the first-wave experiments clearly justify it.
```

---

## Final instruction to the implementation agent

Start by:

1. bootstrapping the environment
2. running `bd init`
3. creating `AGENTS.md`
4. seeding Beads tasks
5. implementing the tokenized dense baseline and PPO/DDP stack

Only then move to routed variants.
