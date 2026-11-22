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
# A second whole GPU node from the GENERAL queue (no reservation): the organizers allow it, it does not count for the
# one-node-per-team limit of the cispahack reservation. It may wait in the general queue before starting.
# The one-job-per-team rule applies to the cispahack reservation only (jureca/hold.sh): several of these may run.
# Do not leave one idle: it spends the shared project budget. Start a queue on it (see jureca/gpu_queue.sh).
#   sbatch -J hold2 --time=07:00:00 jureca/hold_general.sh
# Release it when done:  scancel <jobid>
echo "holding $(hostname) since $(date -Is)"
sleep infinity
