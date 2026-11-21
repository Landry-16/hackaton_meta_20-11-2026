#!/bin/bash
# Run several unlearn_distill.py configs in parallel inside the running "hold" job (one GPU each, 4 max).
# Waits for the hold job to be RUNNING first, so it can be started right after `sbatch -J hold jureca/hold.sh`.
# From the repo root on a JURECA login node, detached:
#   nohup unlearning/hold_sweep.sh "--conf-min 0.02 --conf-max 0.2" "--conf-min 0.05 --conf-max 0.5 --tau 1.5" \
#     > logs/hold_sweep.out 2>&1 &
# Each config gets its own log: logs/hold-<config words joined>.out. Common args: COMMON env (default below).
set -uo pipefail
[ $# -ge 1 ] && [ $# -le 4 ] || { echo "give 1 to 4 configs (one quoted string each)" >&2; exit 1; }
common="${COMMON:---lr 1e-4 --epochs 25 --save-every 5}"
mkdir -p logs

until hold=$(squeue -u "$USER" -h -n hold -t RUNNING -o %i | head -1) && [ -n "$hold" ]; do
  squeue -u "$USER" -h -n hold -o %i | grep -q . || { echo "no hold job queued: sbatch -J hold jureca/hold.sh" >&2; exit 1; }
  echo "$(date +%H:%M:%S) waiting for hold job to start"; sleep 60
done
echo "$(date +%H:%M:%S) hold job $hold running"

gpu=0
for cfg in "$@"; do
  name="hold$(echo " $cfg" | sed -e 's/ --/_/g' -e 's/ /-/g')"
  # shellcheck disable=SC2086
  srun --jobid="$hold" --overlap --ntasks=1 env LAUNCH= REPO_DIR="$PWD" CUDA_VISIBLE_DEVICES=$gpu \
    bash jureca/job.sh unlearning unlearning/unlearn_distill.py $cfg $common > "logs/$name.out" 2>&1 &
  echo "GPU $gpu: $cfg -> logs/$name.out"
  gpu=$((gpu + 1))
done
wait
echo "$(date +%H:%M:%S) all done"
for f in logs/hold_*.out; do echo "== $f"; grep "clean:" "$f" | tail -1 | cut -c1-120; done
