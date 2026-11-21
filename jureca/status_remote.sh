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
hold=$(squeue -u "$USER" -h -n hold -t R -o %i | head -1)
[ -z "$hold" ] && hold=$(squeue -u "$USER" -h -t R -o %i | head -1)
echo "${B}Your node${N} ${D}($USER)${N}"
if [ -n "$hold" ]; then
  node=$(squeue -j "$hold" -h -o %N)
  echo "  job $hold on $node"
  srun --jobid="$hold" --overlap --ntasks=1 nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null < /dev/null |
  while IFS=', ' read -r i u mu mt; do
    n=$((u / 5)); g=""; for j in $(seq 1 20); do [ "$j" -le "$n" ] && g+="#" || g+="."; done
    [ "$u" -gt 0 ] && col=$G || col=$D
    printf "  GPU %s ${col}[%s] %3s%%${N}  memory %5s / %s MiB\n" "$i" "$g" "$u" "$mu" "$mt"
  done
  steps=$(squeue -s -j "$hold" -h -o "%i %j %M" | grep -v '\.batch\|\.extern\|^ *$')
  [ -n "$steps" ] && { echo "  ${D}running steps:${N}"; echo "$steps" | sed 's/^/    /'; }
else
  echo "  ${D}no running job for $USER${N}"
fi
echo

# ---- team GPU queue (jureca/gpu_queue.sh on the team node)
Q=/p/scratch/training2666/savarimuthu1/teambuilders/queue
now=$(date +%s)
what() {   # short description of a queued command: script name + its arguments
  grep '^CMD=' "$1" | sed -e 's/\\ / /g' -e 's/^CMD=//' | grep -oE '[a-z_]+\.py.*|python .*|bash .*' | sed -e 's#^.*/##' | cut -c1-70
}
lastline() { grep -v -i -E 'warn|Fetching|Model loaded|state_dict|^\s*$|^\s+(warnings|checkpoint|return|state)' "$1" 2>/dev/null | tail -1 | cut -c1-95; }
echo "${B}Team GPU queue${N} ${D}(one node for the team; submit with: bash jureca/gpu_submit.sh \"cmd\")${N}"
echo "  ${G}running${N}"
any=0
for j in "$Q"/running/*.job; do [ -e "$j" ] || continue
  n=$(basename "$j" .job); any=1
  owner=$(grep '^OWNER=' "$j" | cut -d= -f2)
  gpu=$(head -1 "$Q/running/$n.log" 2>/dev/null | grep -oE 'GPU [0-9]+')
  el=$(( (now - $(stat -c %Y "$j")) / 60 ))
  printf "    %-13s %-6s %3s min  %-44s\n" "$owner" "${gpu:-GPU ?}" "$el" "$(what "$j")"
  printf "      ${D}%s${N}\n" "$(lastline "$Q/running/$n.log")"
done
[ "$any" = 0 ] && echo "    ${D}nothing running${N}"
echo "  ${Y}waiting${N} ${D}(starts in this order; teammates' jobs go before Landry's)${N}"
k=0
for j in $(ls -tr "$Q"/pending/*.job 2>/dev/null); do
  k=$((k + 1)); [ "$k" -gt 12 ] && { echo "    ... $(ls "$Q"/pending/*.job | wc -l) waiting in total"; break; }
  printf "    %2d. %-13s %s\n" "$k" "$(grep '^OWNER=' "$j" | cut -d= -f2)" "$(what "$j")"
done
[ "$k" = 0 ] && echo "    ${D}empty${N}"
echo "  ${D}finished (last 6)${N}"
ls -t "$Q"/done/*.job 2>/dev/null | head -n 6 | while read -r j; do
  n=$(basename "$j" .job); l="$Q/done/$n.log"
  rc=$(grep -oE 'exit code [0-9]+' "$l" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
  [ "${rc:-x}" = 0 ] && col=$G || col=$R
  res=$(grep -E 'held-out|wrote|share|Score|Traceback|Error' "$l" 2>/dev/null | tail -1 | cut -c1-70)
  printf "    ${col}%-4s${N} %-13s %-40s ${D}%s${N}\n" "${rc:-?}" "$(grep '^OWNER=' "$j" | cut -d= -f2)" "$(what "$j" | cut -c1-40)" "$res"
done
echo
echo "${D}scratch inodes (whole hackathon): $(df --output=iused,iavail /p/scratch/training2666 | tail -1 | awk '{printf "%d used, %d free", $1, $2}')${N}"
echo "${D}full log of a job: bash jureca/gpu_submit.sh log <name>   (names: ls $Q/running $Q/pending)${N}"
