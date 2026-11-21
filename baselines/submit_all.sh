#!/bin/bash
# First valid submission on each task, from a laptop (the ranking needs a score on all three).
# Needs: export HACK_API_KEY=..., data/model.pt and data/test.zip (see README "First submissions").
# Spends no model equality query. Each task then has its cooldown (30 min for model equality).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HACK_API_KEY:?export HACK_API_KEY first}"
run() { uv run -q --with requests --with numpy --with pillow --with torch python "$@"; }

echo "== model equality: all different"
run model_equality/task_template.py
run model_equality/submission.py submission.csv

echo "== unlearning: original model unchanged"
run unlearning/submission.py data/model.pt

echo "== watermark: all-zeros masks"
run baselines/watermark_zeros.py data/test.zip data/watermark_zeros.pt
run -c "import sys; sys.path.insert(0, 'watermark'); from submission_template import submit; submit('data/watermark_zeros.pt')"
