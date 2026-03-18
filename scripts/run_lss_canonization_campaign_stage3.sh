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
print(reports["stage2_json"])
print(campaign["stage_roots"]["stage3_fairness"])
print(reports["stage3_report"])
print(reports["stage3_csv"])
print(reports["stage3_json"])
print(campaign["existing_lane_roots"]["post_pass_b"]["root"])
print(campaign["stage_roots"]["stage2_new_block_base"])
PY
)

STAGE2_JSON="${cfg[0]}"
STAGE3_ROOT="${cfg[1]}"
REPORT_OUTPUT="${cfg[2]}"
REPORT_CSV="${cfg[3]}"
REPORT_JSON="${cfg[4]}"
POST_PASS_B_ROOT="${cfg[5]}"
POST_PASS_C_ROOT="${cfg[6]}"

declare -a specs=()
mapfile -t spec_rows < <(
  source .venv/bin/activate
  python - "$CAMPAIGN_CONFIG" "$STAGE2_JSON" "$POST_PASS_B_ROOT" "$POST_PASS_C_ROOT" "$STAGE3_ROOT" <<'PY'
import json
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
stage2 = json.loads(Path(sys.argv[2]).read_text())
post_pass_b_root = Path(sys.argv[3])
post_pass_c_root = Path(sys.argv[4])
stage3_root = Path(sys.argv[5])
students = campaign["students"]
for candidate in stage2.get("advancing_candidates", []):
    template = campaign["candidates"][candidate]["template"]
    for lane, teacher_root in [("post_pass_b", post_pass_b_root), ("post_pass_c", post_pass_c_root)]:
        seeds = campaign["seed_groups"]["hard_block_existing"]["seeds"] if lane == "post_pass_b" else campaign["seed_groups"]["hard_block_new"]["seeds"]
        for seed in seeds:
            seed_root = teacher_root / f"seed_{seed}"
            for variant in ("sare", "token_dense", "single_expert"):
                output_label = students[variant]["output_label"]
                spec_path = stage3_root / candidate / lane / f"seed_{seed}" / "configs" / f"{output_label}.yaml"
                output_dir = stage3_root / candidate / lane / f"seed_{seed}" / output_label
                print("\t".join([candidate, lane, str(seed), variant, str(template), str(seed_root), str(spec_path), str(output_dir)]))
PY
)

for row in "${spec_rows[@]}"; do
  IFS=$'\t' read -r candidate lane seed variant template seed_root spec_path output_dir <<<"$row"
  render_lss_spec "$CAMPAIGN_CONFIG" "$template" "$variant" "$seed_root" "$seed_root" "$candidate" "$lane" "$seed" "$spec_path" "$output_dir"
  record_command "${output_dir}/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$DEVICE"
  if [[ ! -f "${output_dir}/summary.json" ]]; then
    specs+=("$spec_path")
  fi
done

run_specs_parallel "${specs[@]}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_canonization_campaign stage3-report \
  --stage2-json "$STAGE2_JSON" \
  --stage3-root "$STAGE3_ROOT" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
