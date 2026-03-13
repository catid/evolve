# Tiny Overfit Report

## Protocol

- Environment: `MiniGrid-Empty-5x5-v0`
- Wrapper: fully observed
- Seeds: `1`
- Envs: `1`
- Controls:
  - `flat_dense`
  - `token_dense`
- PPO settings:
  - `rollout_steps=32`
  - `total_updates=400`
  - `update_epochs=8`
  - `minibatches=1`
  - `learning_rate=1e-4`
  - `anneal_lr=false`
  - `clip_coef=0.1`
  - `value_clip_coef=0.1`
  - `ent_coef=0.0`

## Result

- `flat_dense` overfits cleanly:
  - greedy eval return `0.955`
  - greedy eval success `1.000`
  - eval episode length `5.0`
- `token_dense` also overfits cleanly:
  - greedy eval return `0.955`
  - greedy eval success `1.000`
  - eval episode length `5.0`

## Interpretation

The PPO loop is capable of driving nontrivial success on an easy MiniGrid task once truncation handling, seeding, and evaluation semantics are corrected. The repo is no longer blocked on the question “can the trainer learn anything at all?”
