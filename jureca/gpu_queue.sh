#!/bin/bash
# Team GPU queue: ONE node for the whole team (organizers' rule), shared through a folder.
# Runs inside the lead's hold job (as a step) and starts the commands dropped in $TEAM_DIR/queue/pending.
# Several jobs share a GPU (the watermark jobs are CPU-bound and use about 25 % of a card):
#   - lead jobs: up to $SLOTS per GPU (default 3), if the card has less than $MEM_CAP MiB in use;
#   - teammates' jobs: a GPU for themselves, and they go first: while one waits, the least loaded GPU takes no new
#     lead job, so it drains within one job length.
#   start (lead, from the login node, in the repo clone):
#     setsid nohup srun --jobid=<hold id> --overlap --ntasks=1 bash jureca/gpu_queue.sh > /dev/null 2>&1 < /dev/null &
#   GPUs the queue may use: $TEAM_DIR/queue/gpus (default "0 1 2 3"), or GPU_LIST. Submit with jureca/gpu_submit.sh.
# Borrow mode (BORROW_FROM=<other queue dir>): this instance uses GPUs that belong to another queue only while that
# queue leaves them idle, and gives them back at once: as soon as a job waits there or runs on the card, its own
# jobs on that card are killed and put back in pending. Two instances may share one queue folder (mv is the lock).
Q="${QUEUE_DIR:-${TEAM_DIR:?set TEAM_DIR}/queue}"     # QUEUE_DIR: a second queue (the lead runs his own on his GPUs)
LEAD="${QUEUE_LEAD:-savarimuthu1}"
SLOTS="${SLOTS:-3}"
MEM_CAP="${MEM_CAP:-28000}"
mkdir -p "$Q/pending" "$Q/running" "$Q/done"
exec >> "$Q/daemon.log" 2>&1
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-5}"
declare -A PID JOB OWN GPU_OF          # keyed by slot id "gpu.slot"
echo "$(date +%H:%M:%S) queue daemon on $(hostname), $SLOTS lead jobs per GPU"

preempt() {   # $1 = slot: kill the job and requeue it
  pkill -TERM -P "${PID[$1]}" 2>/dev/null; kill "${PID[$1]}" 2>/dev/null; wait "${PID[$1]}" 2>/dev/null
  rm -f "$Q/running/${JOB[$1]}.log"; mv "$Q/running/${JOB[$1]}.job" "$Q/pending/" 2>/dev/null
  echo "$(date +%H:%M:%S) gave GPU ${GPU_OF[$1]} back, requeued ${JOB[$1]}"
  unset "PID[$1]" "JOB[$1]" "OWN[$1]" "GPU_OF[$1]"
}

owner_on() {  # $1 = gpu: does the owning queue run a job on it
  local l
  for l in "$BORROW_FROM"/running/*.log; do [ -e "$l" ] && head -1 "$l" | grep -q "GPU $1," && return 0; done
  return 1
}

mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' || echo 0; }

start() {   # $1 = job file, $2 = gpu, $3 = owner
  local name slot s
  name=$(basename "$1" .job)
  for s in $(seq 1 "$SLOTS"); do [ -z "${PID[$2.$s]:-}" ] && { slot="$2.$s"; break; }; done
  [ -n "${slot:-}" ] || return 1
  mv "$1" "$Q/running/$name.job" 2>/dev/null || return 1
  (
    # shellcheck disable=SC1090
    source "$Q/running/$name.job"         # OWNER, WORKDIR, CMD written by gpu_submit.sh
    echo "owner $OWNER, GPU $2, $(hostname), start $(date -Is)"; echo "cd $WORKDIR && $CMD"; echo
    cd "$WORKDIR" && CUDA_VISIBLE_DEVICES=$2 exec bash -c "$CMD"
  ) > "$Q/running/$name.log" 2>&1 &
  PID[$slot]=$!; JOB[$slot]=$name; OWN[$slot]=$3; GPU_OF[$slot]=$2
  echo "$(date +%H:%M:%S) started $name on GPU $2 ($3)"
}

while true; do
  gpus=${GPU_LIST:-$(cat "$Q/gpus" 2>/dev/null)}; gpus=${gpus:-0 1 2 3}
  for slot in "${!PID[@]}"; do                                   # reap finished jobs
    kill -0 "${PID[$slot]}" 2>/dev/null && continue
    wait "${PID[$slot]}"; rc=$?
    echo "exit code $rc at $(date -Is)" >> "$Q/running/${JOB[$slot]}.log"
    mv "$Q/running/${JOB[$slot]}".* "$Q/done/" 2>/dev/null
    echo "$(date +%H:%M:%S) done ${JOB[$slot]} on GPU ${GPU_OF[$slot]} (exit $rc)"
    unset "PID[$slot]" "JOB[$slot]" "OWN[$slot]" "GPU_OF[$slot]"
  done
  declare -A n=() foreign=()
  for g in $gpus; do n[$g]=0; foreign[$g]=0; done
  if [ -n "${BORROW_FROM:-}" ]; then                             # give borrowed GPUs back when their owner needs them
    owner_waits=0; ls "$BORROW_FROM"/pending/*.job > /dev/null 2>&1 && owner_waits=1
    for g in $gpus; do
      if [ "$owner_waits" = 1 ] || owner_on "$g"; then
        foreign[$g]=1
        for slot in "${!PID[@]}"; do [ "${GPU_OF[$slot]}" = "$g" ] && preempt "$slot"; done
      fi
    done
  fi
  for slot in "${!PID[@]}"; do
    g=${GPU_OF[$slot]}; n[$g]=$(( ${n[$g]:-0} + 1 )); [ "${OWN[$slot]}" != "$LEAD" ] && foreign[$g]=1
  done
  waiting=0                                                      # a teammate job that found no empty GPU
  for f in $(ls -tr "$Q"/pending/*.job 2>/dev/null); do
    owner=$(grep -m1 '^OWNER=' "$f" | cut -d= -f2)
    if [ "$owner" != "$LEAD" ]; then
      placed=0
      for g in $gpus; do
        if [ "${n[$g]}" = 0 ]; then start "$f" "$g" "$owner" && { n[$g]=1; foreign[$g]=1; placed=1; }; break; fi
      done
      [ "$placed" = 0 ] && waiting=1
      continue
    fi
    best=""; drain=""
    for g in $gpus; do                                           # least loaded GPU without a teammate job
      [ "${foreign[$g]}" = 1 ] && continue
      { [ -z "$drain" ] || [ "${n[$g]}" -lt "${n[$drain]}" ]; } && drain=$g
    done
    for g in $gpus; do
      [ "${foreign[$g]}" = 1 ] && continue
      [ "${n[$g]}" -ge "$SLOTS" ] && continue
      [ "$waiting" = 1 ] && [ "$g" = "$drain" ] && continue      # keep it draining for the teammate
      { [ -z "$best" ] || [ "${n[$g]}" -lt "${n[$best]}" ]; } && best=$g
    done
    [ -n "$best" ] || continue
    [ "${n[$best]}" -gt 0 ] && [ "$(mem_used "$best")" -ge "$MEM_CAP" ] && continue
    start "$f" "$best" "$owner" && n[$best]=$(( ${n[$best]} + 1 ))
  done
  sleep "${POLL:-10}"
done
