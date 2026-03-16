# Greedy Recovery Reproduction Note

## Scope

This note reproduces the current DoorKey calibration baseline in the experiment lane used by the published greedy-recovery artifacts:

- `flat_dense`
- recovered `token_dense` (`ent_coef=0.001`)
- `single_expert` (`ent_coef=0.001`)
- `SARE` (`ent_coef=0.001`)

The goal was to verify that the current repo state still supports the accepted starting point for the greedy-recovery phase before adding new checkpoint or sharpening experiments.

## Exact Command

All runs used the repo `uv` environment and all visible GPUs on this machine.

```bash
source .venv/bin/activate
PSMN_OUTPUT_ROOT=outputs/reproductions/greedy_recovery_baseline \
PSMN_EVAL_EPISODES=32 \
./scripts/run_sare_fair_retest.sh
```

That script resolved to:

- world size: `4`
- seed: `7`
- device: `auto` -> CUDA on all visible GPUs
- torch: `2.12.0.dev20260312+cu130`
- git commit: `056a09a7734caa2e6de1a373c81989f696ac1501`
- git dirty: `true`

Dirty-state note:

- the worktree was dirty because this phase had already updated Beads state and markdown link hygiene before reproduction
- no code or config behavior was changed before the reproduction run itself

## Configs Used

- [configs/experiments/minigrid_doorkey_flat_dense_ent1e3.yaml](../../configs/experiments/minigrid_doorkey_flat_dense_ent1e3.yaml)
- [configs/experiments/minigrid_doorkey_token_dense_ent1e3.yaml](../../configs/experiments/minigrid_doorkey_token_dense_ent1e3.yaml)
- [configs/experiments/minigrid_doorkey_single_expert_ent1e3.yaml](../../configs/experiments/minigrid_doorkey_single_expert_ent1e3.yaml)
- [configs/experiments/minigrid_doorkey_sare_ent1e3.yaml](../../configs/experiments/minigrid_doorkey_sare_ent1e3.yaml)

## Output Root

- [outputs/reproductions/greedy_recovery_baseline/report.md](../reproductions/greedy_recovery_baseline/report.md)

Per-run checkpoints:

- [flat_dense latest.pt](../reproductions/greedy_recovery_baseline/minigrid_doorkey_flat_dense_ent1e3/latest.pt)
- [token_dense latest.pt](../reproductions/greedy_recovery_baseline/minigrid_doorkey_token_dense_ent1e3/latest.pt)
- [single_expert latest.pt](../reproductions/greedy_recovery_baseline/minigrid_doorkey_single_expert_ent1e3/latest.pt)
- [sare latest.pt](../reproductions/greedy_recovery_baseline/minigrid_doorkey_sare_ent1e3/latest.pt)

## Reproduced Metrics

| Variant | Greedy Success | Best Sampled Success | Best Sampled Mode | Train Return | Throughput FPS |
| --- | ---: | ---: | --- | ---: | ---: |
| `flat_dense` | `1.0000` | `1.0000` | `sampled_t1.0` | `0.9602` | `9164.3` |
| recovered `token_dense` | `0.7500` | `1.0000` | `sampled_t1.0` | `0.9416` | `6000.1` |
| `single_expert` | `0.0000` | `0.7500` | `sampled_t1.0` | `0.2908` | `6496.4` |
| `SARE` | `0.0000` | `1.0000` | `sampled_t1.0` | `0.7443` | `5736.6` |

## Comparison To Published Calibration Artifacts

Reference artifact:

- [outputs/experiments/sare_retest/report.md](../experiments/sare_retest/report.md)

Result:

- `flat_dense` still solves greedy DoorKey.
- recovered `token_dense` still reaches clearly nonzero greedy DoorKey success.
- `single_expert` is still sampled-competent but greedy-failing.
- `SARE` is still sampled-competent but greedy-failing.

The reproduced metrics are effectively the same as the published calibration baseline for the purposes of this phase. That is enough to justify moving on to checkpoint dynamics and architecture-neutral sharpening, rather than debugging reproducibility first.
