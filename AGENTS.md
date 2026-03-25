# AGENTS.md

## Project Purpose

This repository implements Packet-Switched Morphogenic Network (PSMN) reinforcement learning experiments for small games. The goal is to test whether routed, token-based computation improves over dense tokenized controls under matched or compute-normalized budgets.

Active research questions:

1. Does one-hop routing beat a tokenized dense control?
2. Does adaptive multi-hop routing help on compositional tasks?
3. Does selective relational computation help when object interactions matter?
4. Does persistent option-style routing help on partial observability or sparse reward tasks?
5. Are routed paths sparse, stable, and interpretable enough to justify later morphogenic work?

## Hard Constraints

- Python `3.12` only.
- Use `uv venv` for environment management.
- Install the latest PyTorch nightly for `cu130`.
- Use Beads for task tracking and initialize it with `bd init`.
- Use all visible GPUs by default with `torchrun` + `DistributedDataParallel`.
- Prefer small falsifiable MiniGrid experiments before broader benchmarks.
- Keep this file current when the project workflow or experiment conventions change.

## Issue Tracking

This project uses **bd (beads)** for issue tracking.
Run `bd prime` for workflow context.

Quick reference:

- `bd ready`
- `bd list`
- `bd show <id>`
- `bd update <id> --status in_progress`
- `bd comments add <id> "progress update"`
- `bd close <id> --reason "done"`
- `bd sync --flush-only`

Rules:

- Create a Beads issue before starting substantial work.
- Claim work with `bd update <id> --status in_progress`.
- Use dependencies and parents to model blockers and milestones.
- Close tasks with a short evidence-backed summary.
- Use the `bd` CLI only. Do not edit Beads storage by hand.
- For git pushes on this machine, prefer the current forwarded SSH agent over local keys:
  if `SSH_AUTH_SOCK` is stale, locate the newest `/tmp/ssh-*/agent*` socket, confirm it with
  `ssh-add -L` or `ssh -T git@github.com`, and use that socket for `git pull --rebase`, `bd sync`,
  and `git push`. Do not fall back to stale local GitHub keys unless explicitly requested.

## Initial Experimental Variants

Implement and maintain these first-wave variants:

1. Flattened dense baseline
2. Tokenized dense baseline
3. Tokenized single-expert baseline
4. SARE
5. TREG-H
6. SRW
7. POR

Available recurrent control:

- `token_gru` is a fair PPO recurrent control only on configs that enable sequence-aware rollout batching and minibatching via `ppo.sequence_minibatches: true`.
- If a task/config does not use the sequence-aware PPO path yet, keep `token_gru` in the diagnostic-only lane.

## Experiment Conventions

- Phase 1 environments: MiniGrid DoorKey, KeyCorridor, Memory, DynamicObstacles.
- Sanity tier environments: MiniGrid Empty and FourRooms.
- Phase 2 environments: Procgen CoinRun, Heist, Maze.
- Keep `configs/baseline/` and the original ablation scripts as short smoke/falsification artifacts.
- Canonical control-recovery configs now live under:
  - `configs/diagnostic/` for tiny overfit, sanity-tier, fully observed, and memory-probe checks
  - `configs/experiments/` for longer-run control baselines and routed reruns only after controls are competent
- Procgen support is optional and currently gated by availability of a Python 3.12 compatible Gymnasium Procgen package.
- Use `./scripts/install_procgen_port.sh` for the pinned local Procgen port validated in this repo.
- PPO is the first RL algorithm. Keep interfaces open for IMPALA later.
- Every routed model must be compared against flattened dense, tokenized dense, and tokenized single-expert controls.
- Do not rerun routed models on a target task unless at least one relevant control shows nonzero evaluation performance there.
- Do not treat `Memory` or other persistence-relevant tasks as fair control-vs-routing benchmarks unless the control being compared has a valid recurrent mechanism under the sequence-aware PPO batching path.
- Save configs, summaries, and metrics for every run under `outputs/`.
- Favor simple inspectable PyTorch implementations over framework-heavy abstractions.

## Coding Conventions

- Keep model code under `src/psmn_rl/models/`.
- Keep routing logic isolated in `src/psmn_rl/models/routing/`.
- Keep PPO and rollout logic isolated in `src/psmn_rl/rl/`.
- Prefer dataclasses, explicit tensor shapes, and readable control flow.
- Add shape assertions at key interfaces.
- Add short docstrings for nontrivial modules.
- Avoid custom kernels and premature optimization in the first wave.

## Reproducibility Rules

- Record seed, rank, world size, torch version, CUDA version, and visible GPU count.
- Save the resolved config next to each run.
- Rank 0 alone writes checkpoints and final summaries.
- Evaluation is treated as a deterministic measurement path: rank 0 evaluates, metrics are broadcast, and checkpoints can resume from saved optimizer/update/RNG state.
- Seed each DDP rank deterministically from a shared base seed.
- Vector envs use `SAME_STEP` autoreset semantics; time-limit truncations must be bootstrapped from `final_obs`.
- Keep smoke tests small enough to run on CPU or a single GPU quickly.
- Do not claim a gain unless dense and tokenized controls were run under comparable settings.
- If greedy eval is flat but train success is high, also check sampled eval before judging the control as broken.

## Metrics To Log

Always log:

- episode return
- success rate
- environment steps
- wall-clock
- throughput
- GPU utilization when available
- active parameter or compute proxy
- expert load histogram
- route entropy
- path entropy
- top 10 / 20 / 50 path coverage
- average hop count
- average halting probability
- relational widget usage rate
- option duration
- option switch rate
- seed variance across runs
- episode length
- explained variance

## Before Finalizing Results

- Confirm the relevant Beads issue exists and is updated.
- Run the focused test or smoke command for the changed surface area.
- Save the config and run summary under `outputs/`.
- Write down known limitations or missing controls.
- Prefer fixing control solvability and evaluation correctness over adding routed-model novelty.
- Prefer the `configs/diagnostic/` and `configs/experiments/` lanes for recovery work; leave `configs/baseline/` intact as the smoke lane.
- Prefer another small falsification run over expanding scope.

## Deferred Work

Do not implement these in the first wave unless the controls justify it:

- morphogenic split / merge / prune / compress
- delayed message buffers across time
- unified attention-as-widget systems
- custom Triton kernels
- large benchmark suites
