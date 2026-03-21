#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_expansion_mega_program/campaign.yaml}"

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
  local lane="$3"
  local seed="$4"
  local target_path="$5"
  local output_dir="$6"
  source .venv/bin/activate
  python - "$campaign_config" "$candidate" "$lane" "$seed" "$target_path" "$output_dir" <<'PY'
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

campaign_path, candidate, lane, seed, target_path, output_dir = sys.argv[1:]
campaign = load_campaign_config(Path(campaign_path))
meta = campaign["candidates"][candidate]
student = campaign["students"]["sare"]
raw = yaml.safe_load(Path(meta["template"]).read_text()) or {}
deep_merge(raw, meta.get("overrides", {}))
lane_roots = campaign["lane_roots"][lane]
teacher_root = Path(lane_roots["teacher_root"])
student_root = Path(lane_roots["sare_student_root"])
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
import sys
from pathlib import Path
from psmn_rl.analysis.campaign_config import load_campaign_config

campaign = load_campaign_config(Path(sys.argv[1]))
print(campaign["stage_roots"]["stage1_screening"])
print(campaign["reports"]["stage1_report"])
print(campaign["reports"]["stage1_csv"])
print(campaign["reports"]["stage1_raw_json"])
print(campaign["reports"]["stage1_json"])
for block in campaign["blocks"]["dev"]:
    print(f"block:{block['lane']}:{','.join(str(seed) for seed in block['seeds'])}")
for candidate, meta in campaign["candidates"].items():
    print(f"candidate:{candidate}:{meta.get('stage1_reuse_candidate', '')}")
for root in campaign.get("reuse_roots", {}).get("stage1_sare", []):
    print(f"reuse:{root}")
PY
)

STAGE1_ROOT="${cfg[0]}"
REPORT_OUTPUT="${cfg[1]}"
REPORT_CSV="${cfg[2]}"
REPORT_JSON_RAW="${cfg[3]}"
REPORT_JSON_EFFECTIVE="${cfg[4]}"
ROWS=("${cfg[@]:5}")

mkdir -p "$STAGE1_ROOT"
declare -A BLOCK_SEEDS=()
declare -a CANDIDATES=()
declare -a REUSE_ROOTS=()
declare -A REUSE_CANDIDATES=()
for row in "${ROWS[@]}"; do
  if [[ "$row" == block:* ]]; then
    lane="${row#block:}"
    lane="${lane%%:*}"
    BLOCK_SEEDS["$lane"]="${row##*:}"
  elif [[ "$row" == reuse:* ]]; then
    REUSE_ROOTS+=("${row#reuse:}")
  else
    candidate_payload="${row#candidate:}"
    candidate="${candidate_payload%%:*}"
    alias_candidate="${candidate_payload#${candidate}:}"
    CANDIDATES+=("$candidate")
    if [[ "$alias_candidate" != "$candidate_payload" && -n "$alias_candidate" ]]; then
      REUSE_CANDIDATES["$candidate"]="$alias_candidate"
    fi
  fi
done

declare -a specs=()
for candidate in "${CANDIDATES[@]}"; do
  for lane in "${!BLOCK_SEEDS[@]}"; do
    IFS=',' read -r -a seeds <<<"${BLOCK_SEEDS[$lane]}"
    for seed in "${seeds[@]}"; do
      spec_path="${STAGE1_ROOT}/${candidate}/${lane}/seed_${seed}/configs/kl_lss_sare.yaml"
      output_dir="${STAGE1_ROOT}/${candidate}/${lane}/seed_${seed}/kl_lss_sare"
      render_spec "$CAMPAIGN_CONFIG" "$candidate" "$lane" "$seed" "$spec_path" "$output_dir"
      alias_candidate="${REUSE_CANDIDATES[$candidate]:-}"
      if [[ ! -e "${output_dir}" && -n "$alias_candidate" ]]; then
        alias_dir="${STAGE1_ROOT}/${alias_candidate}/${lane}/seed_${seed}/kl_lss_sare"
        if [[ -f "${alias_dir}/summary.json" ]]; then
          cp -al "${alias_dir}" "${output_dir}"
        fi
      fi
      if [[ ! -e "${output_dir}" ]]; then
        for reuse_root in "${REUSE_ROOTS[@]}"; do
          reuse_dir="${reuse_root}/${candidate}/${lane}/seed_${seed}/kl_lss_sare"
          if [[ -f "${reuse_dir}/summary.json" ]]; then
            cp -al "${reuse_dir}" "${output_dir}"
            break
          fi
        done
      fi
      if [[ ! -e "${output_dir}" && -n "$alias_candidate" ]]; then
        for reuse_root in "${REUSE_ROOTS[@]}"; do
          reuse_dir="${reuse_root}/${alias_candidate}/${lane}/seed_${seed}/kl_lss_sare"
          if [[ -f "${reuse_dir}/summary.json" ]]; then
            cp -al "${reuse_dir}" "${output_dir}"
            break
          fi
        done
      fi
      if [[ ! -f "${output_dir}/summary.json" ]]; then
        record_command "${output_dir}/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$DEVICE"
      fi
      if [[ ! -f "${output_dir}/summary.json" ]]; then
        specs+=("$spec_path")
      fi
    done
  done
done

run_specs_parallel "${specs[@]}"

python -m psmn_rl.analysis.lss_successor_migration stage1-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON_RAW"

cp "$REPORT_JSON_RAW" "$REPORT_JSON_EFFECTIVE"
