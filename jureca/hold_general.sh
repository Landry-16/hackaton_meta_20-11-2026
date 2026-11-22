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
# DO NOT USE during the hackathon. A node from the general queue (no reservation). On 22 Nov 03:40 the organizers
# wrote that such jobs do not count for the one-node limit; at 06:17 an organizer asked us to cancel them: ONE node
# per team in total. All were cancelled at 06:23. Kept only as a record of what was run.
echo "holding $(hostname) since $(date -Is)"
sleep infinity
