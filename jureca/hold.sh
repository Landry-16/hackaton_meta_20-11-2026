#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=64
#SBATCH --partition=dc-gpu
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%j.out
#
# Keep one whole GPU node (4 A100) for the night, so later runs start at once instead of queueing.
# Organizers: ONE node per person at a time (pending jobs count). Never submit a second job while this one exists.
#   sbatch -J hold jureca/hold.sh       then jureca/run.sh uses it automatically while it runs.
# Release it when done:  scancel <jobid>
echo "holding $(hostname) since $(date -Is)"
sleep infinity
