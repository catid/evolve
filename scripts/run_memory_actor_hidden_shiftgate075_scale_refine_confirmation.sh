#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

ROOT="outputs/experiments/memory_actor_hidden_shiftgate075_scale_refine_confirmation"
REPORT_MD="outputs/reports/memory_actor_hidden_shiftgate075_scale_refine_confirmation.md"
REPORT_JSON="outputs/reports/memory_actor_hidden_shiftgate075_scale_refine_confirmation.json"
TMP_DIR="$ROOT/eval_rows"

rm -rf "$ROOT"
mkdir -p "$TMP_DIR" "$(dirname "$REPORT_MD")"

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" -lt 1 ]]; then
  GPU_COUNT=1
fi

declare -a RUN_LABELS=(
  "por_base"
  "por_actor_hidden_partial_shift22"
  "por_actor_hidden_partial_shift22_shiftgate075"
  "por_actor_hidden_partial_shift22_shiftgate075_scale325"
)

declare -a RUN_DIRS=(
  "outputs/experiments/memory_actor_hidden_shiftgate075_scale_refine_probe/memory_por_switchy_rerun"
  "outputs/experiments/memory_actor_hidden_shiftgate075_scale_refine_probe/memory_por_switchy_actor_hidden_partial_shift22"
  "outputs/experiments/memory_actor_hidden_shiftgate075_scale_refine_probe/memory_por_switchy_actor_hidden_partial_shift22_shiftgate075"
  "outputs/experiments/memory_actor_hidden_shiftgate075_scale_refine_probe/memory_por_switchy_actor_hidden_partial_shift22_shiftgate075_scale325"
)

jobs=()
for idx in "${!RUN_LABELS[@]}"; do
  label="${RUN_LABELS[$idx]}"
  run_dir="${RUN_DIRS[$idx]}"
  gpu=$((idx % GPU_COUNT))
  out_json="$TMP_DIR/${label}.json"
  CUDA_VISIBLE_DEVICES="$gpu" ./.venv/bin/python - "$label" "$run_dir" "$out_json" <<'PY' &
import json
import sys
from pathlib import Path
from typing import Any

import torch

from psmn_rl.config import load_config
from psmn_rl.envs.registry import make_vector_env
from psmn_rl.logging import configure_logging
from psmn_rl.models.factory import build_model
from psmn_rl.rl.distributed.ddp import cleanup_distributed, init_distributed
from psmn_rl.rl.ppo.algorithm import collect_policy_diagnostics
from psmn_rl.utils.seed import set_seed

label = sys.argv[1]
run_dir = Path(sys.argv[2])
out_json = Path(sys.argv[3])

MODES = [
    ("greedy", True, 1.0),
    ("lower_t005", False, 0.05),
    ("gap_t0055", False, 0.055),
    ("shoulder_t008", False, 0.08),
]


def load_last_scalar(path: Path) -> dict[str, float]:
    metrics_path = path / "metrics.jsonl"
    last: dict[str, Any] = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar":
            last = row
    return {key: float(value) for key, value in last.items() if isinstance(value, (int, float))}


ctx = init_distributed("auto", "auto")
configure_logging(ctx.is_main_process)
try:
    config = load_config(run_dir / "resolved_config.yaml")
    config.system.device = "auto"
    config.logging.tensorboard = False
    set_seed(config.seed + ctx.rank, deterministic=config.system.deterministic)
    envs = make_vector_env(config.env, seed=config.seed, world_rank=ctx.rank)
    model = build_model(config.model, envs.single_observation_space, envs.single_action_space).to(ctx.device)
    envs.close()
    checkpoint = torch.load(run_dir / "latest.pt", map_location=ctx.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    train_metrics = load_last_scalar(run_dir)
    row: dict[str, Any] = {
        "label": label,
        "run_dir": str(run_dir),
        "train/episode_return": float(train_metrics.get("train/episode_return", 0.0)),
        "train/success_rate": float(train_metrics.get("train/success_rate", 0.0)),
        "policy/option_hidden_film_gate_signal_mean": float(train_metrics.get("policy/option_hidden_film_gate_signal_mean", 0.0)),
        "policy/option_hidden_scale_gate_signal_mean": float(train_metrics.get("policy/option_hidden_scale_gate_signal_mean", 0.0)),
        "policy/option_hidden_shift_gate_signal_mean": float(train_metrics.get("policy/option_hidden_shift_gate_signal_mean", 0.0)),
        "policy/option_hidden_film_gate_mean": float(train_metrics.get("policy/option_hidden_film_gate_mean", 0.0)),
        "policy/option_hidden_scale_gate_power": float(train_metrics.get("policy/option_hidden_scale_gate_power", 0.0)),
        "policy/option_hidden_shift_gate_power": float(train_metrics.get("policy/option_hidden_shift_gate_power", 0.0)),
        "policy/option_hidden_film_scale_norm": float(train_metrics.get("policy/option_hidden_film_scale_norm", 0.0)),
        "policy/option_hidden_film_shift_norm": float(train_metrics.get("policy/option_hidden_film_shift_norm", 0.0)),
    }
    for mode_name, greedy, temperature in MODES:
        diagnostics = collect_policy_diagnostics(
            config=config,
            model=model,
            ctx=ctx,
            episodes=256,
            greedy=greedy,
            temperature=temperature,
            trace_limit=0,
        )
        row[f"{mode_name}/success"] = float(diagnostics.metrics.get("eval_success_rate", 0.0))
        row[f"{mode_name}/return"] = float(diagnostics.metrics.get("eval_return", 0.0))
        row[f"{mode_name}/margin"] = float(diagnostics.metrics.get("eval/action_logit_margin", 0.0))
finally:
    cleanup_distributed(ctx)

out_json.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
PY
  jobs+=("$!")
done

for job in "${jobs[@]}"; do
  wait "$job"
done

./.venv/bin/python - "$TMP_DIR" "$REPORT_MD" "$REPORT_JSON" <<'PY'
import json
import sys
from pathlib import Path

rows_dir = Path(sys.argv[1])
report_md = Path(sys.argv[2])
report_json = Path(sys.argv[3])

rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(rows_dir.glob("*.json"))]
report_json.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")


def row(label: str) -> dict:
    return next(item for item in rows if item["label"] == label)


por_base = row("por_base")
por_shift22 = row("por_actor_hidden_partial_shift22")
por_shift075 = row("por_actor_hidden_partial_shift22_shiftgate075")
por_scale325 = row("por_actor_hidden_partial_shift22_shiftgate075_scale325")

lines = [
    "# Memory ShiftGate075 Scale-Refine Confirmation",
    "",
    "- follow-up to the 64-episode screen in `memory_actor_hidden_shiftgate075_scale_refine_probe.md`",
    "- task: `MiniGrid-MemoryS9-v0`",
    "- evaluation episodes per mode: `256`",
    "- matched rerun roots: base POR, fresh-root `partial_shift22`, confirmed `shiftgate075`, and surviving `shiftgate075_scale325`",
    "",
    "## Aggregate",
    "",
    "| Label | Greedy | Lower t0.05 | Gap t0.055 | Shoulder t0.08 | Greedy Margin | Gap Margin | Shoulder Margin | Train Return | Train Success | Gate Signal | Scale Gate Signal | Shift Gate Signal | Gate Mean | Scale Gate Power | Shift Gate Power | Scale Norm | Shift Norm |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for item in (por_base, por_shift22, por_shift075, por_scale325):
    lines.append(
        f"| `{item['label']}` | `{item['greedy/success']:.4f}` | `{item['lower_t005/success']:.4f}` | `{item['gap_t0055/success']:.4f}` | `{item['shoulder_t008/success']:.4f}` | "
        f"`{item['greedy/margin']:.4f}` | `{item['gap_t0055/margin']:.4f}` | `{item['shoulder_t008/margin']:.4f}` | "
        f"`{item['train/episode_return']:.4f}` | `{item['train/success_rate']:.4f}` | `{item['policy/option_hidden_film_gate_signal_mean']:.4f}` | "
        f"`{item['policy/option_hidden_scale_gate_signal_mean']:.4f}` | `{item['policy/option_hidden_shift_gate_signal_mean']:.4f}` | `{item['policy/option_hidden_film_gate_mean']:.4f}` | "
        f"`{item['policy/option_hidden_scale_gate_power']:.2f}` | `{item['policy/option_hidden_shift_gate_power']:.2f}` | `{item['policy/option_hidden_film_scale_norm']:.4f}` | `{item['policy/option_hidden_film_shift_norm']:.4f}` |"
    )

lines.extend(
    [
        "",
        "## Deltas",
        "",
        f"- vs `shiftgate075`: `scale325` greedy `{por_scale325['greedy/success'] - por_shift075['greedy/success']:+.4f}`, lower `{por_scale325['lower_t005/success'] - por_shift075['lower_t005/success']:+.4f}`, gap `{por_scale325['gap_t0055/success'] - por_shift075['gap_t0055/success']:+.4f}`, shoulder `{por_scale325['shoulder_t008/success'] - por_shift075['shoulder_t008/success']:+.4f}`",
        f"- vs fresh-root `partial_shift22`: `scale325` greedy `{por_scale325['greedy/success'] - por_shift22['greedy/success']:+.4f}`, lower `{por_scale325['lower_t005/success'] - por_shift22['lower_t005/success']:+.4f}`, gap `{por_scale325['gap_t0055/success'] - por_shift22['gap_t0055/success']:+.4f}`, shoulder `{por_scale325['shoulder_t008/success'] - por_shift22['shoulder_t008/success']:+.4f}`",
        "",
        "## Conclusion",
        "",
    ]
)

if (
    por_scale325["greedy/success"] >= por_shift075["greedy/success"]
    and por_scale325["lower_t005/success"] >= por_shift075["lower_t005/success"]
    and por_scale325["gap_t0055/success"] >= por_shift075["gap_t0055/success"]
    and por_scale325["shoulder_t008/success"] >= por_shift075["shoulder_t008/success"]
    and (
        por_scale325["greedy/success"] > por_shift075["greedy/success"]
        or por_scale325["lower_t005/success"] > por_shift075["lower_t005/success"]
        or por_scale325["gap_t0055/success"] > por_shift075["gap_t0055/success"]
        or por_scale325["shoulder_t008/success"] > por_shift075["shoulder_t008/success"]
    )
):
    lines.append(
        f"- `shiftgate075_scale325` confirms as a strict improvement over the prior local incumbent at `greedy/lower/gap/shoulder = {por_scale325['greedy/success']:.4f} / {por_scale325['lower_t005/success']:.4f} / {por_scale325['gap_t0055/success']:.4f} / {por_scale325['shoulder_t008/success']:.4f}`"
    )
    lines.append(
        "- interpretation: lowering actor-hidden FiLM scale from `0.35` to `0.325` strengthens the restored `shiftgate075` surface rather than just reproducing it"
    )
else:
    lines.append(
        f"- `shiftgate075_scale325` does not confirm as a strict improvement over `shiftgate075`: candidate `{por_scale325['greedy/success']:.4f} / {por_scale325['lower_t005/success']:.4f} / {por_scale325['gap_t0055/success']:.4f} / {por_scale325['shoulder_t008/success']:.4f}` versus incumbent `{por_shift075['greedy/success']:.4f} / {por_shift075['lower_t005/success']:.4f} / {por_shift075['gap_t0055/success']:.4f} / {por_shift075['shoulder_t008/success']:.4f}`"
    )
    lines.append(
        "- interpretation: the 64-episode scale gain was either noise or too weak to survive longer external evaluation"
    )

report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
