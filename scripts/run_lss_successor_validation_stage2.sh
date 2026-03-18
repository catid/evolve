#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_successor_validation/campaign.yaml}"

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
  local seed_root="$4"
  local lane="$5"
  local seed="$6"
  local target_path="$7"
  local output_dir="$8"
  source .venv/bin/activate
  python - "$campaign_config" "$candidate" "$variant" "$seed_root" "$lane" "$seed" "$target_path" "$output_dir" <<'PY'
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

campaign_path, candidate, variant, seed_root, lane, seed, target_path, output_dir = sys.argv[1:]
campaign = yaml.safe_load(Path(campaign_path).read_text()) or {}
meta = campaign["candidates"][candidate]
student = campaign["students"][variant]
raw = yaml.safe_load(Path(meta["template"]).read_text()) or {}
deep_merge(raw, meta.get("overrides", {}))
seed_root = Path(seed_root)
raw["name"] = f"{candidate}_{lane}_seed_{seed}_{student['output_label']}"
raw["output_dir"] = output_dir
raw["teacher"]["config"] = str(seed_root / "configs" / "flat_dense_ent1e3.yaml")
raw["teacher"]["checkpoint"] = str(seed_root / "flat_dense_ent1e3" / "latest.pt")
raw["student"]["config"] = str(seed_root / "configs" / student["config_name"])
raw["student"]["checkpoint"] = str(seed_root / student["run_name"] / "latest.pt")
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
print(campaign["stage_roots"]["stage0_blocks"])
print(campaign["stage_roots"]["stage2_fairness"])
print(campaign["reports"]["stage2_report"])
print(campaign["reports"]["stage2_csv"])
print(campaign["reports"]["stage2_json"])
for block in campaign["fresh_blocks"]["blocks"]:
    print(f"block:{block['lane']}:{','.join(str(seed) for seed in block['seeds'])}")
for candidate in campaign["fairness_candidates"]:
    print(f"candidate:{candidate}")
PY
)

STAGE0_ROOT="${cfg[0]}"
STAGE2_ROOT="${cfg[1]}"
REPORT_OUTPUT="${cfg[2]}"
REPORT_CSV="${cfg[3]}"
REPORT_JSON="${cfg[4]}"
ROWS=("${cfg[@]:5}")

mkdir -p "$STAGE2_ROOT"
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
      seed_root="${STAGE0_ROOT}/${lane}/seed_${seed}"
      for variant in sare token_dense single_expert; do
        output_label=$(source .venv/bin/activate && python - "$CAMPAIGN_CONFIG" "$variant" <<'PY'
import sys
from pathlib import Path
import yaml
campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(campaign["students"][sys.argv[2]]["output_label"])
PY
)
        spec_path="${STAGE2_ROOT}/${candidate}/${lane}/seed_${seed}/configs/${output_label}.yaml"
        output_dir="${STAGE2_ROOT}/${candidate}/${lane}/seed_${seed}/${output_label}"
        render_spec "$CAMPAIGN_CONFIG" "$candidate" "$variant" "$seed_root" "$lane" "$seed" "$spec_path" "$output_dir"
        record_command "${output_dir}/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$DEVICE"
        if [[ ! -f "${output_dir}/summary.json" ]]; then
          specs+=("$spec_path")
        fi
      done
    done
  done
done

run_specs_parallel "${specs[@]}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_successor_validation stage2-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
