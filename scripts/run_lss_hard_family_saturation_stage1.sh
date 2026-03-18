#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_hard_family_saturation/campaign.yaml}"

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

render_lss_spec() {
  local campaign_config="$1"
  local candidate="$2"
  local variant="$3"
  local teacher_root="$4"
  local student_root="$5"
  local lane="$6"
  local seed="$7"
  local target_path="$8"
  local output_dir="$9"
  source .venv/bin/activate
  python - "$campaign_config" "$candidate" "$variant" "$teacher_root" "$student_root" "$lane" "$seed" "$target_path" "$output_dir" <<'PY'
import copy
import sys
from pathlib import Path
import yaml

def deep_merge(base, override):
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)

campaign_path, candidate, variant, teacher_root, student_root, lane, seed, target_path, output_dir = sys.argv[1:]
campaign = yaml.safe_load(Path(campaign_path).read_text()) or {}
candidate_meta = campaign["candidates"][candidate]
student_meta = campaign["students"][variant]
raw = yaml.safe_load(Path(candidate_meta["template"]).read_text()) or {}
deep_merge(raw, candidate_meta.get("overrides", {}))
teacher_root = Path(teacher_root)
student_root = Path(student_root)
raw["name"] = f"{candidate}_{lane}_seed_{seed}_{student_meta['output_label']}"
raw["output_dir"] = output_dir
raw["teacher"]["config"] = str(teacher_root / f"seed_{seed}" / "configs" / "flat_dense_ent1e3.yaml")
raw["teacher"]["checkpoint"] = str(teacher_root / f"seed_{seed}" / "flat_dense_ent1e3" / "latest.pt")
raw["student"]["config"] = str(student_root / f"seed_{seed}" / "configs" / student_meta["config_name"])
raw["student"]["checkpoint"] = str(student_root / f"seed_{seed}" / student_meta["run_name"] / "latest.pt")
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
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
definition = json.loads(Path(campaign["reports"]["definition_json"]).read_text())
reports = campaign["reports"]
print(str(definition.get("definition_ready", False)).lower())
print(campaign["stage_roots"]["stage2_dev"])
print(reports["stage2_report"])
print(reports["stage2_csv"])
print(reports["stage2_json"])
for block in definition["hard_family_dev_blocks"]:
    lane = block["lane"]
    roots = campaign["lane_roots"][lane]
    print(
        f"block:{lane}:{','.join(str(seed) for seed in block['seeds'])}:{roots['teacher_root']}:{roots['sare_student_root']}"
    )
for candidate in campaign["candidates"].keys():
    print(f"candidate:{candidate}")
PY
)

DEFINITION_READY="${cfg[0]}"
STAGE1_ROOT="${cfg[1]}"
REPORT_OUTPUT="${cfg[2]}"
REPORT_CSV="${cfg[3]}"
REPORT_JSON="${cfg[4]}"
ROWS=("${cfg[@]:5}")

if [[ "$DEFINITION_READY" != "true" ]]; then
  echo "hard-family definition is not ready; rerun Stage 0 or inspect outputs/reports/hard_family_saturation_definition.md" >&2
  exit 1
fi

mkdir -p "$STAGE1_ROOT"
declare -A BLOCK_SEEDS=()
declare -A BLOCK_TEACHER_ROOT=()
declare -A BLOCK_STUDENT_ROOT=()
declare -a CANDIDATES=()
for row in "${ROWS[@]}"; do
  if [[ "$row" == block:* ]]; then
    lane="${row#block:}"
    lane="${lane%%:*}"
    rest="${row#block:${lane}:}"
    seeds="${rest%%:*}"
    rest="${rest#${seeds}:}"
    teacher_root="${rest%%:*}"
    student_root="${rest#${teacher_root}:}"
    BLOCK_SEEDS["$lane"]="$seeds"
    BLOCK_TEACHER_ROOT["$lane"]="$teacher_root"
    BLOCK_STUDENT_ROOT["$lane"]="$student_root"
  else
    candidate="${row#candidate:}"
    CANDIDATES+=("$candidate")
  fi
done

declare -a specs=()
for candidate in "${CANDIDATES[@]}"; do
  for lane in "${!BLOCK_SEEDS[@]}"; do
    IFS=',' read -r -a seeds <<<"${BLOCK_SEEDS[$lane]}"
    for seed in "${seeds[@]}"; do
      spec_path="${STAGE1_ROOT}/${candidate}/${lane}/seed_${seed}/configs/kl_lss_sare.yaml"
      output_dir="${STAGE1_ROOT}/${candidate}/${lane}/seed_${seed}/kl_lss_sare"
      render_lss_spec "$CAMPAIGN_CONFIG" "$candidate" "sare" "${BLOCK_TEACHER_ROOT[$lane]}" "${BLOCK_STUDENT_ROOT[$lane]}" "$lane" "$seed" "$spec_path" "$output_dir"
      record_command "${output_dir}/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$DEVICE"
      if [[ ! -f "${output_dir}/summary.json" ]]; then
        specs+=("$spec_path")
      fi
    done
  done
done

run_specs_parallel "${specs[@]}"

python -m psmn_rl.analysis.lss_hard_family_saturation stage2-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
