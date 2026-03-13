# DoorKey Reproduction Note

## Scope

This note reproduces the accepted DoorKey comparison in the current experiment lane, not the old smoke lane:

- `flat_dense`
- `token_dense`
- `sare`

The goal was to verify the accepted conclusion before adding new diagnostics:

1. `flat_dense` solves greedy DoorKey.
2. `token_dense` does not.
3. matched `SARE` does not.

## Exact Commands

All three runs used all visible GPUs on this machine (`world_size=4`) and the repo `uv` environment.

```bash
source .venv/bin/activate
NPROC_PER_NODE=$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m psmn_rl.launch \
  --config configs/experiments/minigrid_doorkey_flat_dense.yaml \
  --device auto \
  --output-dir outputs/reproductions/doorkey_controls/flat_dense

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m psmn_rl.launch \
  --config configs/experiments/minigrid_doorkey_token_dense.yaml \
  --device auto \
  --output-dir outputs/reproductions/doorkey_controls/token_dense

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m psmn_rl.launch \
  --config configs/experiments/minigrid_doorkey_sare.yaml \
  --device auto \
  --output-dir outputs/reproductions/doorkey_controls/sare

python -m psmn_rl.analysis.summarize \
  outputs/reproductions/doorkey_controls \
  --output outputs/reproductions/doorkey_controls/report.md \
  --csv outputs/reproductions/doorkey_controls/report.csv
```

All runs used:

- seed: `7`
- world size: `4`
- device: `auto` -> CUDA on all visible GPUs
- torch: `2.12.0.dev20260312+cu130`

## Reproduced Runs

| Variant | Config | Output Dir | Checkpoint | Final Train Return | Final Eval Return | Final Eval Success | Throughput FPS |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `flat_dense` | `configs/experiments/minigrid_doorkey_flat_dense.yaml` | `outputs/reproductions/doorkey_controls/flat_dense` | `outputs/reproductions/doorkey_controls/flat_dense/latest.pt` | `0.9621` | `0.9649` | `1.0000` | `9707.44` |
| `token_dense` | `configs/experiments/minigrid_doorkey_token_dense.yaml` | `outputs/reproductions/doorkey_controls/token_dense` | `outputs/reproductions/doorkey_controls/token_dense/latest.pt` | `0.3100` | `0.0000` | `0.0000` | `6654.92` |
| `sare` | `configs/experiments/minigrid_doorkey_sare.yaml` | `outputs/reproductions/doorkey_controls/sare` | `outputs/reproductions/doorkey_controls/sare/latest.pt` | `0.6386` | `0.0000` | `0.0000` | `6294.76` |

## Comparison To Accepted Published Artifacts

Accepted public artifact roots:

- `outputs/experiments/minigrid_hard_dense/doorkey_flat_dense`
- `outputs/experiments/minigrid_hard_dense/doorkey_token_dense`
- `outputs/experiments/minigrid_sare_doorkey_matched`

High-level comparison:

- `flat_dense` reproduces the accepted greedy DoorKey result exactly on evaluation: solved.
- `token_dense` reproduces the accepted greedy DoorKey result on evaluation: still zero.
- `sare` reproduces the accepted greedy DoorKey result on evaluation: still zero.

Train-side curves do **not** match exactly:

- `flat_dense` stays close on train return and throughput, but value statistics differ.
- `token_dense` and `sare` both show materially higher train return on current `main` than in the accepted artifacts, while still failing greedy evaluation.

## Provenance Caveat

The accepted artifacts are not sufficient to claim exact train-curve reproducibility:

- their `run_meta.json` records `git_commit=59027bb`
- but `59027bb` is only the `.gitignore` / Beads lockfile commit and does **not** contain the `configs/experiments/` lane used by the accepted DoorKey runs
- therefore the saved commit hash alone cannot reconstruct the exact code state that produced those artifacts

The harness bug behind that ambiguity was that `run_meta.json` recorded `git rev-parse HEAD` but not whether the worktree was dirty. That is now fixed in current `main`, so future runs record both `git_commit` and `git_dirty`.

## Conclusion

The accepted **evaluation ranking** reproduces on current `main` and is stable enough for the next phase:

1. `flat_dense` is the best verified greedy DoorKey control.
2. `token_dense` is still not a competent greedy DoorKey control.
3. matched `SARE` is still not a competent greedy DoorKey model.

What does **not** reproduce cleanly is the exact train-side trajectory of the earlier published artifacts. That mismatch is now traced to incomplete provenance metadata in the older runs, not to a contradiction in the accepted evaluation outcome.
