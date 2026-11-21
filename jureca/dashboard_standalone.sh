#!/bin/bash
# JURECA live dashboard for a hackathon team: reservation nodes, team jobs and queue position, GPU use, last log lines.
#
# Run it ON the JURECA login node:      bash dashboard_standalone.sh          (refreshes every 20 s, Ctrl-C to quit)
#                                       bash dashboard_standalone.sh once     (print once)
# Or from your laptop over SSH:         ssh <user>@jureca.fz-juelich.de 'bash -s once' < dashboard_standalone.sh
#
# Fill in the CONFIG block below. Juelich usernames are lowercase (check with: id <username>).

# ============================ CONFIG ============================
RESERVATION="cispahack"          # Slurm reservation of the event
TEAM_USERS=()                     # Juelich usernames, e.g. (alice1 bob2 carol3)
TEAM_NAMES=()                     # display names, same order, e.g. (Alice Bob Carol)
GPU_JOB_NAME=""                   # optional: name of a long-running job whose GPUs to show (e.g. hold); empty = first running job of the first user
LOG_DIRS=()                       # optional: folders with run logs (*.out / *.log) to show the last line of, e.g. ("$HOME/runs/logs")
REFRESH=20                        # seconds between refreshes
# ================================================================

B=$'\e[1m'; D=$'\e[2m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; N=$'\e[0m'

show() {
  echo "${B}JURECA dashboard${N}  ${D}$(date '+%H:%M:%S')${N}"
  echo

  # ---- reservation usage
  total=$(scontrol show res "$RESERVATION" 2>/dev/null | grep -o 'NodeCnt=[0-9]*' | cut -d= -f2)
  run=$(squeue -R "$RESERVATION" -h -t R 2>/dev/null | wc -l)
  pend=$(squeue -R "$RESERVATION" -h -t PD 2>/dev/null | wc -l)
  busy=$(squeue -R "$RESERVATION" -h -t R -o %D 2>/dev/null | awk '{s+=$1} END {print s+0}')
  bar=""
  for i in $(seq 1 "${total:-0}"); do [ "$i" -le "$busy" ] && bar+="${R}#${N}" || bar+="${G}.${N}"; done
  echo "${B}Reservation $RESERVATION${N} [$bar] ${busy}/${total:-?} nodes busy, $run jobs running, ${Y}$pend waiting${N}"
  echo

  # ---- team jobs
  echo "${B}Team jobs${N}"
  if [ "${#TEAM_USERS[@]}" -eq 0 ]; then
    echo "  ${Y}set TEAM_USERS in the CONFIG block${N}"
  fi
  printf "  ${D}%-10s %-10s %-16s %-9s %-10s %s${N}\n" WHO JOBID NAME STATE ELAPSED "NODE / WAIT REASON, START ESTIMATE"
  for k in "${!TEAM_USERS[@]}"; do
    u="${TEAM_USERS[$k]}"; who="${TEAM_NAMES[$k]:-$u}"; found=0
    while read -r id name st el reason start; do
      [ -z "$id" ] && continue; found=1
      if [ "$st" = RUNNING ]; then
        printf "  %-10s %-10s %-16s ${G}%-9s${N} %-10s on %s\n" "$who" "$id" "$name" "$st" "$el" "$reason"
      else
        pos=$(squeue -R "$RESERVATION" -h -t PD --sort=-p -o %i | grep -n "^$id$" | cut -d: -f1)
        printf "  %-10s %-10s %-16s ${Y}%-9s${N} %-10s %s, starts ~%s (queue %s/%s)\n" "$who" "$id" "$name" "$st" "$el" "$reason" "${start/T/ }" "${pos:-?}" "$pend"
      fi
    done < <(squeue -u "$u" -h -o "%i %j %T %M %R %S" 2>/dev/null)
    [ "$found" = 0 ] && printf "  %-10s ${D}no job${N}\n" "$who"
    n=$(squeue -u "$u" -h -o %D 2>/dev/null | awk '{s+=$1} END {print s+0}')
    [ "$n" -gt 1 ] && echo "  ${R}warning: $who holds $n nodes (organizers allow one at a time)${N}"
  done
  echo

  # ---- GPUs of one running job
  me="${TEAM_USERS[0]:-$USER}"
  if [ -n "$GPU_JOB_NAME" ]; then
    job=$(squeue -u "$me" -h -n "$GPU_JOB_NAME" -t R -o %i | head -1)
  else
    job=$(squeue -u "$me" -h -t R -o %i | head -1)
  fi
  echo "${B}GPUs${N} ${D}(job of $me)${N}"
  if [ -n "$job" ] && [ "$me" = "$USER" ]; then
    echo "  job $job on $(squeue -j "$job" -h -o %N)"
    srun --jobid="$job" --overlap --ntasks=1 nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
         --format=csv,noheader,nounits 2>/dev/null |
    while IFS=', ' read -r i u mu mt; do
      g=""; for j in $(seq 1 20); do [ "$j" -le $((u / 5)) ] && g+="#" || g+="."; done
      [ "$u" -gt 0 ] && col=$G || col=$D
      printf "  GPU %s ${col}[%s] %3s%%${N}  memory %5s / %s MiB\n" "$i" "$g" "$u" "$mu" "$mt"
    done
  elif [ -n "$job" ]; then
    echo "  ${D}only $me can read the GPUs of job $job (run the dashboard from that account)${N}"
  else
    echo "  ${D}no running job${N}"
  fi
  echo

  # ---- last line of recent logs
  if [ "${#LOG_DIRS[@]}" -gt 0 ]; then
    echo "${B}Recent logs${N}"
    for d in "${LOG_DIRS[@]}"; do
      for f in $(ls -t "$d"/*.out "$d"/*.log 2>/dev/null | head -6); do
        printf "  %-28s %s\n" "$(basename "$f" | cut -c1-28)" "$(grep -v -i -E 'warn|^\s*$' "$f" | tail -1 | cut -c1-100)"
      done
    done
  fi
}

if [ "${1:-}" = once ]; then show; exit 0; fi
while true; do out="$(show)"; clear; echo "$out"; echo; echo "${D}refresh every ${REFRESH} s, Ctrl-C to quit${N}"; sleep "$REFRESH"; done
