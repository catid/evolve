#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_hard_family_program/campaign.yaml}"

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
  local template="$2"
  local variant="$3"
  local teacher_root="$4"
  local student_root="$5"
  local candidate="$6"
  local lane="$7"
  local seed="$8"
  local target_path="$9"
  local output_dir="${10}"
  source .venv/bin/activate
  python - "$campaign_config" "$template" "$variant" "$teacher_root" "$student_root" "$candidate" "$lane" "$seed" "$target_path" "$output_dir" <<'PY'
import sys
from pathlib import Path
import yaml

campaign_path, template, variant, teacher_root, student_root, candidate, lane, seed, target_path, output_dir = sys.argv[1:]
campaign = yaml.safe_load(Path(campaign_path).read_text()) or {}
student_meta = campaign["students"][variant]
raw = yaml.safe_load(Path(template).read_text()) or {}
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
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
reports = campaign["reports"]
print(campaign["stage_roots"]["stage2_dev"])
print(reports["stage2_report"])
print(reports["stage2_csv"])
print(reports["stage2_json"])
for block in campaign["seed_groups"]["hard_family_dev"]["blocks"]:
    print(f"block:{block['lane']}:{','.join(str(seed) for seed in block['seeds'])}:{campaign['lane_roots'][block['lane']]['teacher_root']}:{campaign['lane_roots'][block['lane']]['sare_student_root']}")
for candidate, meta in campaign["candidates"].items():
    reuse = meta.get("reuse_root", "")
    print(f"candidate:{candidate}:{meta['template']}:{reuse}")
PY
)

STAGE2_ROOT="${cfg[0]}"
REPORT_OUTPUT="${cfg[1]}"
REPORT_CSV="${cfg[2]}"
REPORT_JSON="${cfg[3]}"
ROWS=("${cfg[@]:4}")

mkdir -p "$STAGE2_ROOT"
declare -A BLOCK_SEEDS=()
declare -A BLOCK_TEACHER_ROOT=()
declare -A BLOCK_STUDENT_ROOT=()
declare -a CANDIDATES=()
declare -A CANDIDATE_TEMPLATE=()
declare -A CANDIDATE_REUSE_ROOT=()
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
    candidate="${candidate%%:*}"
    rest="${row#candidate:${candidate}:}"
    template="${rest%%:*}"
    reuse="${rest#${template}:}"
    CANDIDATES+=("$candidate")
    CANDIDATE_TEMPLATE["$candidate"]="$template"
    CANDIDATE_REUSE_ROOT["$candidate"]="$reuse"
  fi
done

declare -a specs=()
for candidate in "${CANDIDATES[@]}"; do
  if [[ -n "${CANDIDATE_REUSE_ROOT[$candidate]}" ]]; then
    continue
  fi
  for lane in "${!BLOCK_SEEDS[@]}"; do
    IFS=',' read -r -a seeds <<<"${BLOCK_SEEDS[$lane]}"
    for seed in "${seeds[@]}"; do
      spec_path="${STAGE2_ROOT}/${candidate}/${lane}/seed_${seed}/configs/kl_lss_sare.yaml"
      output_dir="${STAGE2_ROOT}/${candidate}/${lane}/seed_${seed}/kl_lss_sare"
      render_lss_spec "$CAMPAIGN_CONFIG" "${CANDIDATE_TEMPLATE[$candidate]}" "sare" "${BLOCK_TEACHER_ROOT[$lane]}" "${BLOCK_STUDENT_ROOT[$lane]}" "$candidate" "$lane" "$seed" "$spec_path" "$output_dir"
      record_command "${output_dir}/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$DEVICE"
      if [[ ! -f "${output_dir}/summary.json" ]]; then
        specs+=("$spec_path")
      fi
    done
  done
done

run_specs_parallel "${specs[@]}"

python -m psmn_rl.analysis.lss_hard_family_program stage2-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
