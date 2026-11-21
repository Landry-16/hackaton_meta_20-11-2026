#!/bin/bash
# Team GPU queue: ONE node for the whole team (organizers' rule), shared through a folder.
# Runs inside Landry's hold job (as a step) and starts the commands teammates drop in $TEAM_DIR/queue/pending,
# each on a free GPU, oldest first. A GPU counts as free when nothing of the queue runs on it AND it uses < 1 GB.
#   start (lead, from the login node):
#     srun --jobid=<hold id> --overlap --ntasks=1 bash jureca/gpu_queue.sh > $TEAM_DIR/queue/daemon.log 2>&1 &
#   GPUs the queue may use: $TEAM_DIR/queue/gpus (default "0 1 2 3"), edit it any time.
#   teammates submit with jureca/gpu_submit.sh, see the board / jureca/SETUP.md.
Q="${TEAM_DIR:?set TEAM_DIR}/queue"
mkdir -p "$Q/pending" "$Q/running" "$Q/done"
export HF_HUB_OFFLINE=1
declare -A PID JOB
echo "queue daemon on $(hostname) since $(date -Is)"

gpu_free() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' ')
  [ -n "$used" ] && [ "$used" -lt 1000 ]
}

while true; do
  for g in $(cat "$Q/gpus" 2>/dev/null || echo "0 1 2 3"); do
    p="${PID[$g]:-}"
    if [ -n "$p" ]; then
      kill -0 "$p" 2>/dev/null && continue
      wait "$p"; rc=$?
      echo "exit code $rc at $(date -Is)" >> "$Q/running/${JOB[$g]}.log"
      mv "$Q/running/${JOB[$g]}".* "$Q/done/" 2>/dev/null
      echo "$(date +%H:%M:%S) done ${JOB[$g]} on GPU $g (exit $rc)"
      unset "PID[$g]" "JOB[$g]"
    fi
    next=$(ls -tr "$Q"/pending/*.job 2>/dev/null | head -1)
    [ -z "$next" ] && continue
    gpu_free "$g" || continue
    name=$(basename "$next" .job)
    mv "$next" "$Q/running/$name.job"
    (
      # shellcheck disable=SC1090
      source "$Q/running/$name.job"         # OWNER, WORKDIR, CMD written by gpu_submit.sh
      echo "owner $OWNER, GPU $g, $(hostname), start $(date -Is)"; echo "cd $WORKDIR && $CMD"; echo
      cd "$WORKDIR" && CUDA_VISIBLE_DEVICES=$g exec bash -c "$CMD"
    ) > "$Q/running/$name.log" 2>&1 &
    PID[$g]=$!; JOB[$g]=$name
    echo "$(date +%H:%M:%S) started $name on GPU $g"
  done
  sleep 10
done
