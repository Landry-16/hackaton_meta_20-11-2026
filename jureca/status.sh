#!/bin/bash
# Live JURECA dashboard in your terminal: hackathon nodes, team jobs and queue, GPUs of the hold node, run logs.
#   jureca/status.sh          refresh every 20 s
#   jureca/status.sh once     print once
# Needs the SSH master connection (see jureca/run.sh).
cd "$(dirname "$0")"
host="${JURECA_USER:-savarimuthu1}@jureca.fz-juelich.de"
show() { ssh -S "$HOME/.ssh/cm-jureca" "$host" 'bash -s' < status_remote.sh 2>/dev/null || echo "SSH master connection is closed: reopen it (see jureca/run.sh)"; }
if [ "${1:-}" = once ]; then show; exit; fi
while true; do out="$(show)"; clear; echo "$out"; sleep 20; done
