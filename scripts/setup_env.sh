#!/usr/bin/env bash
# Usage: bash scripts/setup_env.sh [cpu|gpu]
set -euo pipefail
MODE="${1:-cpu}"
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
if [ "$MODE" = "gpu" ]; then pip install -r requirements-gpu.txt; else pip install -r requirements.txt; fi
pip install -e .
python -m spacy download en_core_web_sm
python -c "import benepar; benepar.download('benepar_en3')"
