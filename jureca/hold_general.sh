#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --partition=dc-gpu
#SBATCH --time=08:30:00
#SBATCH --output=logs/%x-%j.out
#
# A whole GPU node from the GENERAL queue. Organizers (Michel, 22 Nov 06:25): nodes outside the reservation are not
# limited, but the cispahack reservation is MAGNETIC: Slurm pulls our jobs onto its nodes even without --reservation,
# and those count for the one-node-per-team limit (3 of our jobs were cancelled for that at 06:23).
# ALWAYS exclude the reserved nodes and check where the job landed:
#   RES=$(scontrol show reservation cispahack | grep -oE "Nodes=[^ ]+" | cut -d= -f2)
#   sbatch --exclude="$RES" --time=04:00:00 -J gen2 jureca/hold_general.sh
#   scontrol show node <node> | grep ReservationName      -> must print nothing, otherwise scancel at once
# Do not leave one idle: it spends the shared project budget. Release with scancel <jobid>.
echo "holding $(hostname) since $(date -Is)"
sleep infinity
