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
print(reports["stage3_json"])
print(campaign["stage_roots"]["stage4_holdout"])
print(reports["stage4_report"])
print(reports["stage4_csv"])
print(reports["stage4_json"])
for block in campaign["seed_groups"]["hard_family_holdout"]["blocks"]:
    print(f"block:{block['lane']}:{','.join(str(seed) for seed in block['seeds'])}:{campaign['lane_roots'][block['lane']]['teacher_root']}:{campaign['lane_roots'][block['lane']]['sare_student_root']}:{campaign['lane_roots'][block['lane']]['token_student_root']}:{campaign['lane_roots'][block['lane']]['single_expert_root']}")
for candidate, meta in campaign["candidates"].items():
    print(f"candidate:{candidate}:{meta['template']}")
PY
)

STAGE3_JSON="${cfg[0]}"
STAGE4_ROOT="${cfg[1]}"
REPORT_OUTPUT="${cfg[2]}"
REPORT_CSV="${cfg[3]}"
REPORT_JSON="${cfg[4]}"
ROWS=("${cfg[@]:5}")

mkdir -p "$STAGE4_ROOT"
BEST_CANDIDATE="$(
  source .venv/bin/activate
  python - "$STAGE3_JSON" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
print(payload.get("best_candidate") or "")
PY
)"

declare -A BLOCK_SEEDS=()
declare -A BLOCK_TEACHER_ROOT=()
declare -A BLOCK_SARE_ROOT=()
declare -A BLOCK_TOKEN_ROOT=()
declare -A BLOCK_SINGLE_ROOT=()
declare -A CANDIDATE_TEMPLATE=()
for row in "${ROWS[@]}"; do
  if [[ "$row" == block:* ]]; then
    lane="${row#block:}"
    lane="${lane%%:*}"
    rest="${row#block:${lane}:}"
    seeds="${rest%%:*}"
    rest="${rest#${seeds}:}"
    teacher_root="${rest%%:*}"
    rest="${rest#${teacher_root}:}"
    sare_root="${rest%%:*}"
    rest="${rest#${sare_root}:}"
    token_root="${rest%%:*}"
    single_root="${rest#${token_root}:}"
    BLOCK_SEEDS["$lane"]="$seeds"
    BLOCK_TEACHER_ROOT["$lane"]="$teacher_root"
    BLOCK_SARE_ROOT["$lane"]="$sare_root"
    BLOCK_TOKEN_ROOT["$lane"]="$token_root"
    BLOCK_SINGLE_ROOT["$lane"]="$single_root"
  else
    candidate="${row#candidate:}"
    candidate="${candidate%%:*}"
    template="${row#candidate:${candidate}:}"
    CANDIDATE_TEMPLATE["$candidate"]="$template"
  fi
done

declare -a specs=()
if [[ -n "$BEST_CANDIDATE" ]]; then
  for lane in "${!BLOCK_SEEDS[@]}"; do
    IFS=',' read -r -a seeds <<<"${BLOCK_SEEDS[$lane]}"
    for seed in "${seeds[@]}"; do
      for variant in sare token_dense single_expert; do
        case "$variant" in
          sare)
            student_root="${BLOCK_SARE_ROOT[$lane]}"
            output_label="kl_lss_sare"
            ;;
          token_dense)
            student_root="${BLOCK_TOKEN_ROOT[$lane]}"
            output_label="kl_lss_token_dense"
            ;;
          single_expert)
            student_root="${BLOCK_SINGLE_ROOT[$lane]}"
            output_label="kl_lss_single_expert"
            ;;
        esac
        spec_path="${STAGE4_ROOT}/${BEST_CANDIDATE}/${lane}/seed_${seed}/configs/${output_label}.yaml"
        output_dir="${STAGE4_ROOT}/${BEST_CANDIDATE}/${lane}/seed_${seed}/${output_label}"
        render_lss_spec "$CAMPAIGN_CONFIG" "${CANDIDATE_TEMPLATE[$BEST_CANDIDATE]}" "$variant" "${BLOCK_TEACHER_ROOT[$lane]}" "$student_root" "$BEST_CANDIDATE" "$lane" "$seed" "$spec_path" "$output_dir"
        record_command "${output_dir}/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$DEVICE"
        if [[ ! -f "${output_dir}/summary.json" ]]; then
          specs+=("$spec_path")
        fi
      done
    done
  done
fi

run_specs_parallel "${specs[@]}"

python -m psmn_rl.analysis.lss_hard_family_program stage4-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage3-json "$STAGE3_JSON" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
