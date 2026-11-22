#!/bin/bash
# From the laptop: sync this repo to JURECA and queue one watermark script on the team node (shared GPU queue).
#   jureca/q.sh <tag> watermark/seg_ext.py --method pixelseal --fold 0 --synth 400
# The log ends up in $TEAM_DIR/queue/done/<name>.log ; jureca/q.sh wait <tag> blocks until all jobs with that tag are done.
set -euo pipefail
cd "$(dirname "$0")/.."
host="${JURECA_USER:-savarimuthu1}@jureca.fz-juelich.de"
ssh_cmd=(ssh -S "$HOME/.ssh/cm-jureca")
if [ "$1" = wait ]; then
  "${ssh_cmd[@]}" "$host" "source ~/.bashrc; Q=\$TEAM_DIR/queue_lead; while ls \$Q/pending \$Q/running | grep -q -- '$2'; do sleep 20; done; for f in \$(ls -tr \$Q/done/*$2*.log); do echo == \$(basename \$f); grep -E 'held-out|wrote|Traceback|Error|exit code' \$f | tail -3; done"
  exit
fi
tag="$1"; shift
rsync -a --exclude data --exclude .git --exclude .board --exclude .venv --exclude .idea --exclude models --exclude release --exclude output --exclude plots \
  -e "${ssh_cmd[*]}" ./ "$host:/p/scratch/training2666/savarimuthu1/teambuilders/code/savarimuthu1/"
"${ssh_cmd[@]}" "$host" "source ~/.bashrc; D=\$TEAM_DIR/Hackathon_Watermark_Localisation_Task; R=\$TEAM_DIR/code/savarimuthu1; cd \$D && QUEUE_DIR=\$TEAM_DIR/queue_lead QTAG=$tag bash \$R/jureca/gpu_submit.sh \"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=\$R/watermark WMLOC_MODELS=\$D/models \$D/.venv/bin/python \$R/$* \" | sed 's/^/[$tag] /'"
