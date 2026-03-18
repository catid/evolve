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
print(reports["stage4_json"])
print(reports["stage5_json"])
print(campaign["frozen_pack"])
print(campaign["stage_roots"]["stage6_retry_pack"])
print(reports["candidate_summary_markdown"])
print(reports["candidate_metrics_json"])
print(reports["combined_report_markdown"])
print(reports["combined_report_csv"])
print(reports["retry_report_markdown"])
print(reports["retry_report_csv"])
print(reports["successor_pack_markdown"])
print(reports["successor_pack_json"])
print(reports["gate_report_markdown"])
print(reports["gate_report_json"])
print(reports["decision_memo"])
print(campaign["historical_lane_roots"]["fresh_final"]["teacher_root"])
print(campaign["historical_lane_roots"]["fresh_final"]["sare_student_root"])
print(campaign["historical_lane_roots"]["fresh_final"]["token_student_root"])
print(campaign["historical_lane_roots"]["fresh_final"]["single_expert_root"])
PY
)

STAGE3_JSON="${cfg[0]}"
STAGE4_JSON="${cfg[1]}"
STAGE5_JSON="${cfg[2]}"
FROZEN_PACK="${cfg[3]}"
STAGE6_ROOT="${cfg[4]}"
SUMMARY_OUTPUT="${cfg[5]}"
METRICS_OUTPUT="${cfg[6]}"
COMBINED_OUTPUT="${cfg[7]}"
COMBINED_CSV="${cfg[8]}"
RETRY_OUTPUT="${cfg[9]}"
RETRY_CSV="${cfg[10]}"
SUCCESSOR_MD="${cfg[11]}"
SUCCESSOR_JSON="${cfg[12]}"
GATE_MD="${cfg[13]}"
GATE_JSON="${cfg[14]}"
DECISION_MEMO="${cfg[15]}"
FRESH_FINAL_TEACHER_ROOT="${cfg[16]}"
FRESH_FINAL_SARE_ROOT="${cfg[17]}"
FRESH_FINAL_TOKEN_ROOT="${cfg[18]}"
FRESH_FINAL_SINGLE_ROOT="${cfg[19]}"

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

if [[ -z "$BEST_CANDIDATE" ]]; then
  mkdir -p "$(dirname "$SUCCESSOR_MD")" "$(dirname "$SUCCESSOR_JSON")" "$(dirname "$GATE_MD")" "$(dirname "$GATE_JSON")"
  cat >"$SUCCESSOR_MD" <<'EOF'
# Canonization Successor Candidate Pack

- status: `not generated`
- reason: `no candidate survived hard-block fairness`
EOF
  cat >"$SUCCESSOR_JSON" <<'EOF'
{
  "status": "not_generated",
  "reason": "no candidate survived hard-block fairness",
  "stage": "Stage 3"
}
EOF
  cat >"$GATE_MD" <<'EOF'
# Canonization Gate Report

- status: `not run`
- reason: `no candidate survived hard-block fairness`
EOF
  cat >"$GATE_JSON" <<'EOF'
{
  "status": "not_run",
  "reason": "no candidate survived hard-block fairness",
  "stage": "Stage 3"
}
EOF
  source .venv/bin/activate
  python -m psmn_rl.analysis.lss_canonization_campaign decision-memo \
    --stage2-json "$(python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml
campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(campaign["reports"]["stage2_json"])
PY
)" \
    --stage3-json "$STAGE3_JSON" \
    --stage4-json "$STAGE4_JSON" \
    --stage5-json "$STAGE5_JSON" \
    --output "$DECISION_MEMO"
  exit 0
fi

declare -a specs=()
mapfile -t spec_rows < <(
  source .venv/bin/activate
  python - "$CAMPAIGN_CONFIG" "$STAGE3_JSON" "$STAGE6_ROOT" "$FRESH_FINAL_TEACHER_ROOT" "$FRESH_FINAL_SARE_ROOT" "$FRESH_FINAL_TOKEN_ROOT" "$FRESH_FINAL_SINGLE_ROOT" <<'PY'
import json
import sys
from pathlib import Path
import yaml

campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
stage3 = json.loads(Path(sys.argv[2]).read_text())
stage6_root = Path(sys.argv[3])
teacher_root = Path(sys.argv[4])
sare_root = Path(sys.argv[5])
token_root = Path(sys.argv[6])
single_root = Path(sys.argv[7])
best = stage3.get("best_candidate")
if not best:
    raise SystemExit(0)
template = campaign["candidates"][best]["template"]
students = campaign["students"]
for seed in campaign["seed_groups"]["retry_block"]["seeds"]:
    for variant, student_root in [("sare", sare_root), ("token_dense", token_root), ("single_expert", single_root)]:
        output_label = students[variant]["output_label"]
        spec_path = stage6_root / best / "fresh_final" / f"seed_{seed}" / "configs" / f"{output_label}.yaml"
        output_dir = stage6_root / best / "fresh_final" / f"seed_{seed}" / output_label
        print("\t".join([best, str(seed), variant, str(template), str(teacher_root / f"seed_{seed}"), str(student_root / f"seed_{seed}"), str(spec_path), str(output_dir)]))
PY
)

for row in "${spec_rows[@]}"; do
  IFS=$'\t' read -r candidate seed variant template teacher_root student_root spec_path output_dir <<<"$row"
  render_lss_spec "$CAMPAIGN_CONFIG" "$template" "$variant" "$teacher_root" "$student_root" "$candidate" "fresh_final" "$seed" "$spec_path" "$output_dir"
  record_command "${output_dir}/command.txt" python -m psmn_rl.analysis.learner_state_supervision run --spec "$spec_path" --device "$DEVICE"
  if [[ ! -f "${output_dir}/summary.json" ]]; then
    specs+=("$spec_path")
  fi
done

run_specs_parallel "${specs[@]}"

source .venv/bin/activate
python -m psmn_rl.analysis.lss_canonization_campaign successor-pack \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --stage3-json "$STAGE3_JSON" \
  --stage4-json "$STAGE4_JSON" \
  --stage5-json "$STAGE5_JSON" \
  --stage6-root "$STAGE6_ROOT" \
  --summary-output "$SUMMARY_OUTPUT" \
  --metrics-output "$METRICS_OUTPUT" \
  --combined-report-output "$COMBINED_OUTPUT" \
  --combined-report-csv "$COMBINED_CSV" \
  --retry-report-output "$RETRY_OUTPUT" \
  --retry-report-csv "$RETRY_CSV" \
  --successor-pack-markdown "$SUCCESSOR_MD" \
  --successor-pack-json "$SUCCESSOR_JSON"

python -m psmn_rl.analysis.claim_gate \
  --frozen-pack "$FROZEN_PACK" \
  --candidate-pack "$SUCCESSOR_JSON" \
  --output "$GATE_MD" \
  --json-output "$GATE_JSON"

python -m psmn_rl.analysis.lss_canonization_campaign decision-memo \
  --stage2-json "$(python - "$CAMPAIGN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml
campaign = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(campaign["reports"]["stage2_json"])
PY
)" \
  --stage3-json "$STAGE3_JSON" \
  --stage4-json "$STAGE4_JSON" \
  --stage5-json "$STAGE5_JSON" \
  --gate-json "$GATE_JSON" \
  --output "$DECISION_MEMO"
