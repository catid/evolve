#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_successor_migration/campaign.yaml}"

gpu_count() {
  source .venv/bin/activate
  python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
}

run_specs_parallel() {
  local -a specs=("$@")
  local count
  count=$(gpu_count)
  if [[ "${#specs[@]}" -eq 0 ]]; then
    return
  fi
  if [[ "$DEVICE" == "cpu" || "$count" -le 0 ]]; then
    for spec in "${specs[@]}"; do
      source .venv/bin/activate
      python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec" --device "$DEVICE"
    done
    return
  fi
  local slot=0
  local active=0
  local failed=0
  for spec in "${specs[@]}"; do
    local gpu=$((slot % count))
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      source .venv/bin/activate
      python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec" --device cuda
    ) &
    slot=$((slot + 1))
    active=$((active + 1))
    if [[ "$active" -ge "$count" ]]; then
      if ! wait -n; then
        failed=1
      fi
      active=$((active - 1))
    fi
  done
  while [[ "$active" -gt 0 ]]; do
    if ! wait -n; then
      failed=1
    fi
    active=$((active - 1))
  done
  [[ "$failed" -eq 0 ]]
}

record_command() {
  local target="$1"
  shift
  mkdir -p "$(dirname "$target")"
  printf '%q ' "$@" >"$target"
  printf '\n' >>"$target"
}

render_spec() {
  local campaign_config="$1"
  local candidate="$2"
  local variant="$3"
  local lane="$4"
  local seed="$5"
  local target_path="$6"
  local output_dir="$7"
  source .venv/bin/activate
  python - "$campaign_config" "$candidate" "$variant" "$lane" "$seed" "$target_path" "$output_dir" <<'PY'
import copy
import sys
from pathlib import Path
import yaml
from psmn_rl.analysis.campaign_config import load_campaign_config

def deep_merge(base, override):
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)

campaign_path, candidate, variant, lane, seed, target_path, output_dir = sys.argv[1:]
campaign = load_campaign_config(Path(campaign_path))
meta = campaign["candidates"][candidate]
student = campaign["students"][variant]
raw = yaml.safe_load(Path(meta["template"]).read_text()) or {}
deep_merge(raw, meta.get("overrides", {}))
lane_roots = campaign["lane_roots"][lane]
teacher_root = Path(lane_roots["teacher_root"])
student_root_key = {
    "sare": "sare_student_root",
    "token_dense": "token_student_root",
    "single_expert": "single_expert_root",
}[variant]
student_root = Path(lane_roots[student_root_key])
teacher_seed_root = teacher_root / f"seed_{seed}"
student_seed_root = student_root / f"seed_{seed}"
raw["name"] = f"{candidate}_{lane}_seed_{seed}_{student['output_label']}"
raw["output_dir"] = output_dir
raw["teacher"]["config"] = str(teacher_seed_root / "configs" / "flat_dense_ent1e3.yaml")
raw["teacher"]["checkpoint"] = str(teacher_seed_root / "flat_dense_ent1e3" / "latest.pt")
raw["student"]["config"] = str(student_seed_root / "configs" / student["config_name"])
raw["student"]["checkpoint"] = str(student_seed_root / student["run_name"] / "latest.pt")
target = Path(target_path)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
PY
}

source .venv/bin/activate
readarray -t cfg < <(
  python - "$CAMPAIGN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path
from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
stage2 = json.loads(Path(campaign["reports"]["stage2_json"]).read_text())
print(campaign["stage_roots"]["stage3_holdout"])
print(campaign["reports"]["stage3_report"])
print(campaign["reports"]["stage3_csv"])
print(campaign["reports"]["stage3_json"])
for block in campaign["blocks"]["holdout"]:
    print(f"block:{block['lane']}:{','.join(str(seed) for seed in block['seeds'])}")
for candidate in stage2["surviving_candidates"]:
    print(f"candidate:{candidate}")
PY
)

STAGE3_ROOT="${cfg[0]}"
REPORT_OUTPUT="${cfg[1]}"
REPORT_CSV="${cfg[2]}"
REPORT_JSON="${cfg[3]}"
ROWS=("${cfg[@]:4}")

mkdir -p "$STAGE3_ROOT"
declare -A BLOCK_SEEDS=()
declare -a CANDIDATES=()
for row in "${ROWS[@]}"; do
  if [[ "$row" == block:* ]]; then
    lane="${row#block:}"
    lane="${lane%%:*}"
    BLOCK_SEEDS["$lane"]="${row##*:}"
  else
    CANDIDATES+=("${row#candidate:}")
  fi
done

declare -a specs=()
for candidate in "${CANDIDATES[@]}"; do
  for lane in "${!BLOCK_SEEDS[@]}"; do
    IFS=',' read -r -a seeds <<<"${BLOCK_SEEDS[$lane]}"
    for seed in "${seeds[@]}"; do
      for variant in sare token_dense single_expert; do
        output_label=$(source .venv/bin/activate && python - "$CAMPAIGN_CONFIG" "$variant" <<'PY'
import sys
from pathlib import Path
from psmn_rl.analysis.campaign_config import load_campaign_config
campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["students"][sys.argv[2]]["output_label"])
PY
)
        spec_path="${STAGE3_ROOT}/${candidate}/${lane}/seed_${seed}/configs/${output_label}.yaml"
        output_dir="${STAGE3_ROOT}/${candidate}/${lane}/seed_${seed}/${output_label}"
        render_spec "$CAMPAIGN_CONFIG" "$candidate" "$variant" "$lane" "$seed" "$spec_path" "$output_dir"
        record_command "${output_dir}/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$DEVICE"
        if [[ ! -f "${output_dir}/summary.json" ]]; then
          specs+=("$spec_path")
        fi
      done
    done
  done
done

run_specs_parallel "${specs[@]}"

python -m psmn_rl.analysis.lss_successor_migration stage3-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
