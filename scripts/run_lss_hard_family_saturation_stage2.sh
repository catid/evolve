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
print(reports["stage2_json"])
print(campaign["stage_roots"]["stage3_fairness"])
print(reports["stage3_report"])
print(reports["stage3_csv"])
print(reports["stage3_json"])
for block in definition["hard_family_dev_blocks"]:
    lane = block["lane"]
    roots = campaign["lane_roots"][lane]
    print(
        f"block:{lane}:{','.join(str(seed) for seed in block['seeds'])}:{roots['teacher_root']}:{roots['sare_student_root']}:{roots['token_student_root']}:{roots['single_expert_root']}"
    )
for candidate in campaign["candidates"].keys():
    print(f"candidate:{candidate}")
PY
)

STAGE1_JSON="${cfg[0]}"
STAGE2_ROOT="${cfg[1]}"
REPORT_OUTPUT="${cfg[2]}"
REPORT_CSV="${cfg[3]}"
REPORT_JSON="${cfg[4]}"
ROWS=("${cfg[@]:5}")

mkdir -p "$STAGE2_ROOT"
declare -A BLOCK_SEEDS=()
declare -A BLOCK_TEACHER_ROOT=()
declare -A BLOCK_SARE_ROOT=()
declare -A BLOCK_TOKEN_ROOT=()
declare -A BLOCK_SINGLE_ROOT=()
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
  fi
done

mapfile -t advancing < <(
  source .venv/bin/activate
  python - "$STAGE1_JSON" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
for candidate in payload.get("advancing_candidates", []):
    print(candidate)
PY
)

declare -a specs=()
for candidate in "${advancing[@]}"; do
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
        spec_path="${STAGE2_ROOT}/${candidate}/${lane}/seed_${seed}/configs/${output_label}.yaml"
        output_dir="${STAGE2_ROOT}/${candidate}/${lane}/seed_${seed}/${output_label}"
        render_lss_spec "$CAMPAIGN_CONFIG" "$candidate" "$variant" "${BLOCK_TEACHER_ROOT[$lane]}" "$student_root" "$lane" "$seed" "$spec_path" "$output_dir"
        record_command "${output_dir}/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$DEVICE"
        if [[ ! -f "${output_dir}/summary.json" ]]; then
          specs+=("$spec_path")
        fi
      done
    done
  done
done

run_specs_parallel "${specs[@]}"

python -m psmn_rl.analysis.lss_hard_family_saturation stage3-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage2-json "$STAGE1_JSON" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
