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
# Organizers: ONE node per person at a time (pending jobs count). Never submit a second job while this one exists.
#   sbatch -J hold jureca/hold.sh       then jureca/run.sh uses it automatically while it runs.
# Release it when done:  scancel <jobid>
echo "holding $(hostname) since $(date -Is)"
sleep infinity
