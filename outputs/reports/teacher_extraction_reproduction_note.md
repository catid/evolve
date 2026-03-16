# Teacher-Extraction Reproduction Note

## Scope

This note records the reproduced DoorKey baseline used for the teacher-guided extraction phase.

Run command:

```bash
source .venv/bin/activate
./scripts/run_teacher_extraction_reproduction.sh
```

Evaluation settings:

- device: `auto`
- world size: `4`
- eval episodes per mode: `32`
- output root: `outputs/reproductions/teacher_extraction_baseline`

Git state recorded in the reproduced run metadata:

- commit: `bbf942e98636063e42aa416f32afecf936e034ab`
- dirty: `true`

## Exact Training Commands

The reproduction script expanded to these training commands:

```bash
torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch \
  --config configs/experiments/minigrid_doorkey_flat_dense_ent1e3.yaml \
  --device auto \
  --output-dir outputs/reproductions/teacher_extraction_baseline/minigrid_doorkey_flat_dense_ent1e3

torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch \
  --config configs/experiments/minigrid_doorkey_token_dense.yaml \
  --device auto \
  --output-dir outputs/reproductions/teacher_extraction_baseline/minigrid_doorkey_token_dense

torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch \
  --config configs/experiments/minigrid_doorkey_token_dense_ent1e3.yaml \
  --device auto \
  --output-dir outputs/reproductions/teacher_extraction_baseline/minigrid_doorkey_token_dense_ent1e3

torchrun --standalone --nproc_per_node=4 -m psmn_rl.launch \
  --config configs/experiments/minigrid_doorkey_sare_ent1e3.yaml \
  --device auto \
  --output-dir outputs/reproductions/teacher_extraction_baseline/minigrid_doorkey_sare_ent1e3
```

Then the script ran:

```bash
python -m psmn_rl.analysis.policy_diagnostics \
  outputs/reproductions/teacher_extraction_baseline \
  --episodes 32 \
  --device auto \
  --group-by run_name \
  --output outputs/reproductions/teacher_extraction_baseline/report.md \
  --csv outputs/reproductions/teacher_extraction_baseline/report.csv
```

## Reproduced Results

Source artifact: [report.md](../reproductions/teacher_extraction_baseline/report.md)

| Run | Config | Checkpoint | Greedy Success | Best Sampled Success | Train Return | Throughput |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `minigrid_doorkey_flat_dense_ent1e3` | `configs/experiments/minigrid_doorkey_flat_dense_ent1e3.yaml` | `outputs/reproductions/teacher_extraction_baseline/minigrid_doorkey_flat_dense_ent1e3/latest.pt` | `1.0000` | `1.0000` | `0.9602` | `9075.1123` |
| `minigrid_doorkey_token_dense` | `configs/experiments/minigrid_doorkey_token_dense.yaml` | `outputs/reproductions/teacher_extraction_baseline/minigrid_doorkey_token_dense/latest.pt` | `0.0000` | `0.1250` | `0.3100` | `6047.1079` |
| `minigrid_doorkey_token_dense_ent1e3` | `configs/experiments/minigrid_doorkey_token_dense_ent1e3.yaml` | `outputs/reproductions/teacher_extraction_baseline/minigrid_doorkey_token_dense_ent1e3/latest.pt` | `0.7500` | `1.0000` | `0.9416` | `6022.3223` |
| `minigrid_doorkey_sare_ent1e3` | `configs/experiments/minigrid_doorkey_sare_ent1e3.yaml` | `outputs/reproductions/teacher_extraction_baseline/minigrid_doorkey_sare_ent1e3/latest.pt` | `0.0000` | `1.0000` | `0.7443` | `5742.9775` |

## Conclusion

The reproduced baseline matched the accepted public artifacts closely enough to justify the teacher-guided extraction work:

- `flat_dense` stayed the strongest greedy DoorKey control
- recovered `token_dense` stayed the canonical tokenized control
- `SARE` stayed sampled-competent but greedy-failing

That reproduced state is the baseline referenced by the teacher-extraction reports in this phase.
