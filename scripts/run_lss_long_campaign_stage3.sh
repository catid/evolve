#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_long_campaign/campaign.yaml}"
STAGE2_JSON="${PSMN_STAGE2_JSON:-outputs/reports/long_campaign_stage2_screening.json}"
MANIFEST_PATH="${PSMN_MANIFEST_PATH:-configs/claims/doorkey_frozen_claim.yaml}"
OUTPUT_ROOT="${PSMN_OUTPUT_ROOT:-outputs/experiments/lss_long_campaign/stage3}"
REPORT_OUTPUT="${PSMN_REPORT_OUTPUT:-outputs/reports/long_campaign_stage3_fairness.md}"
REPORT_CSV="${PSMN_REPORT_CSV:-outputs/reports/long_campaign_stage3_fairness.csv}"
REPORT_JSON="${PSMN_REPORT_JSON:-outputs/reports/long_campaign_stage3_fairness.json}"

mkdir -p "$OUTPUT_ROOT"

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
  local -a pids=()
  for spec in "${specs[@]}"; do
    local gpu=$((slot % count))
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      source .venv/bin/activate
      python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec" --device cuda
    ) &
    pids+=("$!")
    slot=$((slot + 1))
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
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

declare -a specs=()
mapfile -t spec_rows < <(
  source .venv/bin/activate
  python - "$CAMPAIGN_CONFIG" "$STAGE2_JSON" "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
stage2 = json.loads(Path(sys.argv[2]).read_text())
output_root = Path(sys.argv[3])
weak = campaign["seed_groups"]["weak_block"]
lane = str(weak["lane"])
lane_cfg = campaign["lane_roots"][lane]
students = campaign["students"]
for candidate_id in stage2.get("advancing_candidates", []):
    template = str(campaign["candidates"][candidate_id]["template"])
    for variant in ("sare", "token_dense", "single_expert"):
        for seed in weak["seeds"]:
            teacher_root = Path(lane_cfg["teacher_root"]) / f"seed_{seed}"
            if variant == "sare":
                student_root = Path(lane_cfg["sare_student_root"]) / f"seed_{seed}"
            elif variant == "token_dense":
                student_root = Path(lane_cfg["token_student_root"]) / f"seed_{seed}"
            else:
                student_root = Path(lane_cfg["single_expert_root"]) / f"seed_{seed}"
            spec_path = output_root / candidate_id / lane / f"seed_{seed}" / "configs" / f"{students[variant]['output_label']}.yaml"
            output_dir = output_root / candidate_id / lane / f"seed_{seed}" / students[variant]["output_label"]
            print("\t".join([
                candidate_id,
                lane,
                str(seed),
                variant,
                template,
                str(teacher_root),
                str(student_root),
                str(spec_path),
                str(output_dir),
            ]))
PY
)

for row in "${spec_rows[@]}"; do
  IFS=$'\t' read -r candidate lane seed variant template teacher_root student_root spec_path output_dir <<<"$row"
  source .venv/bin/activate
  python - "$CAMPAIGN_CONFIG" "$variant" "$template" "$teacher_root" "$student_root" "$spec_path" "$output_dir" "$candidate" "$seed" <<'PY'
import sys
from pathlib import Path
import yaml

campaign_path, variant, template, teacher_root, student_root, spec_path, output_dir, candidate, seed = sys.argv[1:]
campaign = yaml.safe_load(Path(campaign_path).read_text()) or {}
student_meta = campaign["students"][variant]
raw = yaml.safe_load(Path(template).read_text()) or {}
teacher_root = Path(teacher_root)
student_root = Path(student_root)
raw["name"] = f"seed_{seed}_{candidate}_{student_meta['output_label']}"
raw["output_dir"] = output_dir
raw["teacher"]["config"] = str(teacher_root / "configs" / "flat_dense_ent1e3.yaml")
raw["teacher"]["checkpoint"] = str(teacher_root / "flat_dense_ent1e3" / "latest.pt")
raw["student"]["config"] = str(student_root / "configs" / student_meta["config_name"])
raw["student"]["checkpoint"] = str(student_root / student_meta["run_name"] / "latest.pt")
target = Path(spec_path)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
PY
  record_command "${output_dir}/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$DEVICE"
  specs+=("$spec_path")
done

run_specs_parallel "${specs[@]}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_long_campaign stage3-report \
  --manifest "$MANIFEST_PATH" \
  --stage3-root "$OUTPUT_ROOT" \
  --stage2-json "$STAGE2_JSON" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
