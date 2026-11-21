#!/bin/bash
# Run a command on the TEAM node (one node per team: never sbatch / salloc yourself).
# From the JURECA login node, in the folder where the command must run:
#   bash jureca/gpu_submit.sh "source $TEAM_DIR/unlearning/.venv/bin/activate && python my_script.py --lr 1e-4"
#   bash jureca/gpu_submit.sh status          what is pending, running, done
#   bash jureca/gpu_submit.sh log <name>      follow a job's output
#   bash jureca/gpu_submit.sh cancel <name>   remove a pending job
# The command runs as Landry's account on a free GPU (CUDA_VISIBLE_DEVICES is set for you), with NO internet:
# download weights / packages on the login node first. Write outputs inside $TEAM_DIR.
Q="${TEAM_DIR:?set TEAM_DIR in ~/.bashrc (jureca/SETUP.md)}/queue"
case "${1:-}" in
  ""|-h|--help) sed -n '2,10p' "$0"; exit 0 ;;
  status)
    for s in pending running done; do
      echo "== $s"; ls -tr "$Q/$s"/*.job 2>/dev/null | tail -n "$([ $s = done ] && echo 8 || echo 50)" | while read -r f; do
        n=$(basename "$f" .job); printf "  %-34s %s\n" "$n" "$(tail -n 1 "$Q/$s/$n.log" 2>/dev/null | cut -c1-90)"; done
    done
    echo "== daemon"; tail -n 3 "$Q/daemon.log" 2>/dev/null; exit 0 ;;
  log) f=$(ls "$Q"/running/"$2".log "$Q"/done/"$2".log 2>/dev/null | head -1); [ -n "$f" ] && tail -f "$f" || echo "no log yet for $2"; exit 0 ;;
  cancel) rm -v "$Q/pending/$2.job"; exit 0 ;;
esac
mkdir -p "$Q/pending"
name="$(date +%H%M%S)-$USER-$RANDOM"
printf 'OWNER=%q\nWORKDIR=%q\nCMD=%q\n' "$USER" "$PWD" "$1" > "$Q/pending/$name.job.tmp"
mv "$Q/pending/$name.job.tmp" "$Q/pending/$name.job"
echo "queued $name ($(ls "$Q"/pending/*.job | wc -l) pending). Follow it: bash jureca/gpu_submit.sh log $name"
