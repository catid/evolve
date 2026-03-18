#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_canonization_campaign/campaign.yaml}"

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
raw["teacher"]["config"] = str(teacher_root / "configs" / "flat_dense_ent1e3.yaml")
raw["teacher"]["checkpoint"] = str(teacher_root / "flat_dense_ent1e3" / "latest.pt")
raw["student"]["config"] = str(student_root / "configs" / student_meta["config_name"])
raw["student"]["checkpoint"] = str(student_root / student_meta["run_name"] / "latest.pt")
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
print(campaign["stage_roots"]["stage4_replication"])
print(reports["stage4_report"])
print(reports["stage4_csv"])
print(reports["stage4_json"])
for lane, roots in campaign["historical_lane_roots"].items():
    print(f"{lane}:{roots['teacher_root']}:{roots['sare_student_root']}:{roots['token_student_root']}:{roots['single_expert_root']}")
PY
)

STAGE3_JSON="${cfg[0]}"
STAGE4_ROOT="${cfg[1]}"
REPORT_OUTPUT="${cfg[2]}"
REPORT_CSV="${cfg[3]}"
REPORT_JSON="${cfg[4]}"
ROOT_ROWS=("${cfg[@]:5}")

declare -a specs=()
mapfile -t spec_rows < <(
  source .venv/bin/activate
  python - "$CAMPAIGN_CONFIG" "$STAGE3_JSON" "${ROOT_ROWS[@]}" "$STAGE4_ROOT" <<'PY'
import json
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
stage3 = json.loads(Path(sys.argv[2]).read_text())
root_rows = sys.argv[3:-1]
stage4_root = Path(sys.argv[-1])
best = stage3.get("best_candidate")
if not best:
    raise SystemExit(0)
lane_roots = {}
for row in root_rows:
    lane, teacher_root, sare_root, token_root, single_root = row.split(":", 4)
    lane_roots[lane] = {
        "teacher_root": Path(teacher_root),
        "sare_student_root": Path(sare_root),
        "token_student_root": Path(token_root),
        "single_expert_root": Path(single_root),
    }
template = campaign["candidates"][best]["template"]
students = campaign["students"]
for block in campaign["seed_groups"]["strong_blocks"]["blocks"]:
    lane = block["lane"]
    roots = lane_roots[lane]
    for seed in block["seeds"]:
        for variant, root_key in [("sare", "sare_student_root"), ("token_dense", "token_student_root"), ("single_expert", "single_expert_root")]:
            output_label = students[variant]["output_label"]
            spec_path = stage4_root / best / lane / f"seed_{seed}" / "configs" / f"{output_label}.yaml"
            output_dir = stage4_root / best / lane / f"seed_{seed}" / output_label
            teacher_root = roots["teacher_root"] / f"seed_{seed}"
            student_root = roots[root_key] / f"seed_{seed}"
            print("\t".join([best, lane, str(seed), variant, str(template), str(teacher_root), str(student_root), str(spec_path), str(output_dir)]))
PY
)

for row in "${spec_rows[@]}"; do
  IFS=$'\t' read -r candidate lane seed variant template teacher_root student_root spec_path output_dir <<<"$row"
  render_lss_spec "$CAMPAIGN_CONFIG" "$template" "$variant" "$teacher_root" "$student_root" "$candidate" "$lane" "$seed" "$spec_path" "$output_dir"
  record_command "${output_dir}/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$DEVICE"
  if [[ ! -f "${output_dir}/summary.json" ]]; then
    specs+=("$spec_path")
  fi
done

run_specs_parallel "${specs[@]}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_canonization_campaign stage4-report \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage3-json "$STAGE3_JSON" \
  --stage4-root "$STAGE4_ROOT" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
