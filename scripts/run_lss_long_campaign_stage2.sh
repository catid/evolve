#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_long_campaign/campaign.yaml}"
OUTPUT_ROOT="${PSMN_OUTPUT_ROOT:-outputs/experiments/lss_long_campaign/stage2}"
BASELINE_FINAL_CSV="${PSMN_BASELINE_FINAL_CSV:-outputs/reports/lss_final_block_single_expert_control_report.csv}"
REPORT_OUTPUT="${PSMN_REPORT_OUTPUT:-outputs/reports/long_campaign_stage2_screening.md}"
REPORT_CSV="${PSMN_REPORT_CSV:-outputs/reports/long_campaign_stage2_screening.csv}"
REPORT_JSON="${PSMN_REPORT_JSON:-outputs/reports/long_campaign_stage2_screening.json}"

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
  python - "$CAMPAIGN_CONFIG" "$OUTPUT_ROOT" <<'PY'
import sys
from pathlib import Path
import yaml

campaign_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])
campaign = yaml.safe_load(campaign_path.read_text()) or {}
weak = campaign["seed_groups"]["weak_block"]
lane = str(weak["lane"])
lane_cfg = campaign["lane_roots"][lane]
students = campaign["students"]
for candidate_id, candidate in campaign["candidates"].items():
    template = str(candidate["template"])
    for seed in weak["seeds"]:
        teacher_root = Path(lane_cfg["teacher_root"]) / f"seed_{seed}"
        sare_root = Path(lane_cfg["sare_student_root"]) / f"seed_{seed}"
        spec_path = output_root / candidate_id / lane / f"seed_{seed}" / "configs" / "kl_lss_sare.yaml"
        output_dir = output_root / candidate_id / lane / f"seed_{seed}" / students["sare"]["output_label"]
        print("\t".join([
            candidate_id,
            lane,
            str(seed),
            template,
            str(teacher_root),
            str(sare_root),
            str(spec_path),
            str(output_dir),
        ]))
PY
)

for row in "${spec_rows[@]}"; do
  IFS=$'\t' read -r candidate lane seed template teacher_root sare_root spec_path output_dir <<<"$row"
  mkdir -p "$(dirname "$spec_path")"
  source .venv/bin/activate
  python - "$template" "$teacher_root" "$sare_root" "$spec_path" "$output_dir" "$candidate" "$seed" <<'PY'
import sys
from pathlib import Path
import yaml

template, teacher_root, student_root, spec_path, output_dir, candidate, seed = sys.argv[1:]
raw = yaml.safe_load(Path(template).read_text()) or {}
teacher_root = Path(teacher_root)
student_root = Path(student_root)
raw["name"] = f"seed_{seed}_{candidate}_kl_lss_sare"
raw["output_dir"] = output_dir
raw["teacher"]["config"] = str(teacher_root / "configs" / "flat_dense_ent1e3.yaml")
raw["teacher"]["checkpoint"] = str(teacher_root / "flat_dense_ent1e3" / "latest.pt")
raw["student"]["config"] = str(student_root / "configs" / "sare_ent1e3.yaml")
raw["student"]["checkpoint"] = str(student_root / "sare_ent1e3" / "latest.pt")
target = Path(spec_path)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
PY
  record_command "${output_dir}/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$DEVICE"
  specs+=("$spec_path")
done

run_specs_parallel "${specs[@]}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_long_campaign stage2-report \
  --stage2-root "$OUTPUT_ROOT" \
  --baseline-final-csv "$BASELINE_FINAL_CSV" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
