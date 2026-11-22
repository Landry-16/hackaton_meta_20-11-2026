#!/bin/bash
# Run a command line ON THE TEAM NODE (not on the login node), from the laptop:  jureca/node.sh 'ls $TEAM_DIR'
# Needed when a login node has no /p/scratch mounted (22 Nov 06:35): the compute node always has it.
# NODE_RSYNC is the --rsync-path to give rsync so that the remote side also runs on the node.
JOB="${JURECA_JOB:-15902083}"
NODE_RSYNC="srun --jobid=$JOB --overlap --ntasks=1 -u rsync"
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  ssh -S "$HOME/.ssh/cm-jureca" -o BatchMode=yes "${JURECA_USER:-savarimuthu1}@jureca.fz-juelich.de" \
    "srun --jobid=$JOB --overlap --ntasks=1 -u bash -s" <<< "source ~/.bashrc; $*"
fi
