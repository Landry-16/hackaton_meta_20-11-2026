#!/bin/bash
# From the laptop: copy this repo to your JURECA clone, run one script on a GPU node, wait, print the log.
#   jureca/run.sh watermark watermark/regime.py [args...]
# If a job named "hold" is running (sbatch -J hold jureca/hold.sh), the script runs inside it at once, on GPU $GPU
# (default 0; give each parallel run its own: GPU=1 jureca/run.sh ...). Otherwise it queues a new job.
# Needs the SSH master connection, opened once with your TOTP code:
#   ssh -fN -M -S ~/.ssh/cm-jureca -o ControlPersist=10h -i ~/.ssh/judac-hack <juelich-user>@jureca.fz-juelich.de
set -euo pipefail
cd "$(dirname "$0")/.."
user="${JURECA_USER:-savarimuthu1}"
host="$user@jureca.fz-juelich.de"
ssh_cmd=(ssh -S "$HOME/.ssh/cm-jureca")
remote='$TEAM_DIR/code/$USER'

"${ssh_cmd[@]}" -O check "$host" 2>/dev/null || { echo "no SSH master connection: open it first (see top of this file)" >&2; exit 1; }
rsync -a --exclude data --exclude .git --exclude .board --exclude .venv --exclude .idea --exclude models --exclude release \
  -e "${ssh_cmd[*]}" ./ "$host:/p/scratch/training2666/savarimuthu1/teambuilders/code/$user/"

name="$(basename "${2%.py}")"
hold="$("${ssh_cmd[@]}" "$host" "squeue -u $user -h -n hold -t RUNNING -o %i" | head -1)"
if [ -n "$hold" ]; then
  echo "running in hold job $hold on GPU ${GPU:-0}"
  "${ssh_cmd[@]}" "$host" "source ~/.bashrc; cd $remote && mkdir -p logs && srun --jobid=$hold --overlap --ntasks=1 env LAUNCH= HF_HUB_OFFLINE=1 REPO_DIR=\$PWD CUDA_VISIBLE_DEVICES=${GPU:-0} bash jureca/job.sh $* 2>&1 | tee logs/$name-hold-\$(date +%H%M%S).out | grep -v -i -E 'warn|Fetching|Model loaded|state_dict'"
  exit
fi
# organizers' rule: at most ONE node per person at a time (pending jobs count), or jobs get cancelled and access revoked
mine="$("${ssh_cmd[@]}" "$host" "squeue -u $user -h -o '%i %j %T'")"
if [ -n "$mine" ]; then
  echo "refused: you already have a job on JURECA (one node per person at a time):" >&2
  echo "$mine" >&2
  echo "wait for it, or run inside it (sbatch -J hold jureca/hold.sh)." >&2
  exit 1
fi
"${ssh_cmd[@]}" "$host" "source ~/.bashrc; cd $remote && mkdir -p logs && j=\$(sbatch --parsable -J $name jureca/job.sh $*) && echo job \$j && while squeue -h -j \$j | grep -q .; do sleep 10; done; grep -v -i -E 'warn|Fetching|Model loaded' logs/$name-\$j.out; grep -i -E 'error|traceback' -A5 logs/$name-\$j.err | tail -20"
