#!/bin/bash
# Prints the team dashboard. Runs on the JURECA login node (jureca/status.sh sends it over SSH).
B=$'\e[1m'; D=$'\e[2m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; C=$'\e[36m'; N=$'\e[0m'
TEAM=(savarimuthu1 billiouw1 ravelonahina1)
NAMES=(Landry Nathan Mirado)
WM=/p/scratch/training2666/savarimuthu1/teambuilders/Hackathon_Watermark_Localisation_Task

echo "${B}JURECA dashboard${N}  ${D}$(date '+%H:%M:%S')  (refreshes every 20 s, Ctrl-C to quit)${N}"
echo

# ---- the hackathon reservation
total=$(scontrol show res cispahack | grep -o 'NodeCnt=[0-9]*' | cut -d= -f2)
run=$(squeue -R cispahack -h -t R | wc -l)
pend=$(squeue -R cispahack -h -t PD | wc -l)
busy=$(squeue -R cispahack -h -t R -o %D | paste -sd+ - | bc 2>/dev/null)
bar=""; for i in $(seq 1 "${total:-30}"); do [ "$i" -le "${busy:-0}" ] && bar+="${R}#${N}" || bar+="${G}.${N}"; done
echo "${B}Hackathon nodes${N} [$bar] ${busy:-0}/${total} busy   ${run} jobs running, ${Y}${pend} waiting${N}"
echo

# ---- our team's jobs
echo "${B}Team jobs${N}"
printf "  ${D}%-8s %-10s %-14s %-9s %-10s %s${N}\n" WHO JOBID NAME STATE ELAPSED "WAIT REASON / START ESTIMATE"
any=0
for k in "${!TEAM[@]}"; do
  while read -r id name st el reason start; do
    [ -z "$id" ] && continue; any=1
    if [ "$st" = RUNNING ]; then col=$G; info="on $reason"
    else col=$Y; pos=$(squeue -R cispahack -h -t PD --sort=-p -o %i | grep -n "^$id$" | cut -d: -f1)
         info="$reason, starts ~${start/T/ }  (queue position ${pos:-?}/$pend)"; fi
    printf "  %-8s %-10s %-14s ${col}%-9s${N} %-10s %s\n" "${NAMES[$k]}" "$id" "$name" "$st" "$el" "$info"
  done < <(squeue -u "${TEAM[$k]}" -h -o "%i %j %T %M %R %S" 2>/dev/null | awk '{if($4=="RUNNING"){print $1,$2,$3,$4,$5,"-"}else{print $1,$2,$3,$4,$5,$6}}' | awk '{print $1,$2,$3,$4,$5,$6}' | sed 's/ RUNNING / RUNNING /')
done
[ "$any" = 0 ] && echo "  ${D}none${N}"
echo

# ---- GPUs of the hold node
hold=$(squeue -u savarimuthu1 -h -n hold -t R -o %i | head -1)
echo "${B}Landry's hold node${N}"
if [ -n "$hold" ]; then
  node=$(squeue -j "$hold" -h -o %N)
  echo "  job $hold on $node"
  srun --jobid="$hold" --overlap --ntasks=1 nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null |
  while IFS=', ' read -r i u mu mt; do
    n=$((u / 5)); g=""; for j in $(seq 1 20); do [ "$j" -le "$n" ] && g+="#" || g+="."; done
    [ "$u" -gt 0 ] && col=$G || col=$D
    printf "  GPU %s ${col}[%s] %3s%%${N}  memory %5s / %s MiB\n" "$i" "$g" "$u" "$mu" "$mt"
  done
  steps=$(squeue -s -j "$hold" -h -o "%i %j %M" | grep -v '\.batch\|\.extern\|^ *$')
  [ -n "$steps" ] && { echo "  ${D}running steps:${N}"; echo "$steps" | sed 's/^/    /'; }
else
  echo "  ${Y}not running yet${N} (GPU runs queue as normal jobs until it starts)"
fi
echo

# ---- watermark progress
echo "${B}Watermark progress${N} ${D}(last line of each log)${N}"
for f in "$WM"/output/synth_*.log; do [ -f "$f" ] && printf "  %-22s %s\n" "$(basename "$f" .log)" "$(grep -E 'samples|Error|error' "$f" | tail -1)"; done
for f in $(ls -t /p/scratch/training2666/savarimuthu1/teambuilders/code/savarimuthu1/logs/*.out 2>/dev/null | head -5); do
  printf "  %-22s %s\n" "$(basename "$f" .out | cut -c1-22)" "$(grep -v -i -E 'warn|Fetching|Model loaded|state_dict|^\s*$' "$f" | tail -1 | cut -c1-110)"
done
