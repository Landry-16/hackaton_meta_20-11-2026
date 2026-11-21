#!/bin/bash
# From the laptop: copy this repo to your JURECA clone, run one script on a GPU node, wait, print the log.
#   jureca/run.sh watermark watermark/regime.py [args...]
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
"${ssh_cmd[@]}" "$host" "source ~/.bashrc; cd $remote && mkdir -p logs && j=\$(sbatch --parsable -J $name jureca/job.sh $*) && echo job \$j && while squeue -h -j \$j | grep -q .; do sleep 10; done; grep -v -i -E 'warn|Fetching|Model loaded' logs/$name-\$j.out; grep -i -E 'error|traceback' -A5 logs/$name-\$j.err | tail -20"
