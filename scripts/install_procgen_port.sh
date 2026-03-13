#!/usr/bin/env bash
set -euo pipefail

VENDOR_DIR="${1:-.vendor/procgen-gymnasium}"
REPO_URL="https://github.com/Achronus/procgen-gymnasium"
PINNED_COMMIT="d8473639048149ed8965653da3f6460b6b5e2224"

mkdir -p "$(dirname "$VENDOR_DIR")"

if [[ ! -d "$VENDOR_DIR/.git" ]]; then
  git clone "$REPO_URL" "$VENDOR_DIR"
fi

git -C "$VENDOR_DIR" fetch --all --tags
git -C "$VENDOR_DIR" checkout "$PINNED_COMMIT"

python - <<'PY' "$VENDOR_DIR/pyproject.toml"
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
needle = 'requires-python = ">=3.13"'
replacement = 'requires-python = ">=3.12"'
if needle in text:
    path.write_text(text.replace(needle, replacement))
PY

source .venv/bin/activate
uv pip install -e "$VENDOR_DIR"

python - <<'PY'
import gymnasium as gym
import procgen_gym

env = gym.make("procgen_gym/procgen-coinrun-v0", distribution_mode="easy", start_level=0, num_levels=1)
obs, info = env.reset(seed=7)
print({"shape": getattr(obs, "shape", None), "action_space": str(env.action_space)})
env.close()
PY
