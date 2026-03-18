#!/usr/bin/env bash
set -euo pipefail

DEVICE="${PSMN_DEVICE:-auto}"
CAMPAIGN_CONFIG="${PSMN_CAMPAIGN_CONFIG:-configs/experiments/lss_long_campaign/campaign.yaml}"
STAGE3_JSON="${PSMN_STAGE3_JSON:-outputs/reports/long_campaign_stage3_fairness.json}"
STAGE3_CSV="${PSMN_STAGE3_CSV:-outputs/reports/long_campaign_stage3_fairness.csv}"
MANIFEST_PATH="${PSMN_MANIFEST_PATH:-configs/claims/doorkey_frozen_claim.yaml}"
BASELINE_COMBINED_CSV="${PSMN_BASELINE_COMBINED_CSV:-outputs/reports/lss_frozen_claim_updated_combined_doorkey_report.csv}"
OUTPUT_ROOT="${PSMN_OUTPUT_ROOT:-outputs/experiments/lss_long_campaign/stage4}"
REPORT_OUTPUT="${PSMN_REPORT_OUTPUT:-outputs/reports/long_campaign_stage4_replication.md}"
REPORT_CSV="${PSMN_REPORT_CSV:-outputs/reports/long_campaign_stage4_replication.csv}"
REPORT_JSON="${PSMN_REPORT_JSON:-outputs/reports/long_campaign_stage4_replication.json}"

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
  python - "$CAMPAIGN_CONFIG" "$STAGE3_JSON" "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
stage3 = json.loads(Path(sys.argv[2]).read_text())
best_candidate = stage3.get("best_candidate")
if not best_candidate:
    raise SystemExit(0)
output_root = Path(sys.argv[3])
students = campaign["students"]
for group_name in ("original", "fresh", "fresh_extra"):
    group = campaign["seed_groups"][group_name]
    lane = str(group["lane"])
    lane_cfg = campaign["lane_roots"][lane]
    for variant in ("sare", "token_dense", "single_expert"):
        for seed in group["seeds"]:
            teacher_root = Path(lane_cfg["teacher_root"]) / f"seed_{seed}"
            if variant == "sare":
                student_root = Path(lane_cfg["sare_student_root"]) / f"seed_{seed}"
            elif variant == "token_dense":
                student_root = Path(lane_cfg["token_student_root"]) / f"seed_{seed}"
            else:
                student_root = Path(lane_cfg["single_expert_root"]) / f"seed_{seed}"
            template = str(campaign["candidates"][best_candidate]["template"])
            spec_path = output_root / best_candidate / lane / f"seed_{seed}" / "configs" / f"{students[variant]['output_label']}.yaml"
            output_dir = output_root / best_candidate / lane / f"seed_{seed}" / students[variant]["output_label"]
            print("\t".join([
                best_candidate,
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
python -m psmn_rl.analysis.lss_long_campaign stage4-report \
  --manifest "$MANIFEST_PATH" \
  --baseline-combined-csv "$BASELINE_COMBINED_CSV" \
  --stage3-csv "$STAGE3_CSV" \
  --stage3-json "$STAGE3_JSON" \
  --stage4-root "$OUTPUT_ROOT" \
  --output "$REPORT_OUTPUT" \
  --csv "$REPORT_CSV" \
  --json "$REPORT_JSON"
