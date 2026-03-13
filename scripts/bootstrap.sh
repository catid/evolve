#!/usr/bin/env bash
set -euo pipefail

uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install -U pip setuptools wheel
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
uv pip install -e '.[dev]'

echo "Optional Procgen support: ./scripts/install_procgen_port.sh"
