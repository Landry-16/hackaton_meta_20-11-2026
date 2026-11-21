#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Run one script of this repo on a GPU node, inside the task's shared data folder and venv.
# From the repo root on JURECA:
#   mkdir -p logs
#   sbatch -J unl-base jureca/job.sh unlearning unlearning/task_template.py
#   sbatch -J wm-base  jureca/job.sh watermark watermark/submission_template.py --data release --out submission.pt
# Needs TEAM_DIR in ~/.bashrc (see jureca/SETUP.md). Logs land in logs/<job name>-<id>.out
set -euo pipefail

task="$1"; shift
case "$task" in
  watermark)      data_folder="Hackathon_Watermark_Localisation_Task" ;;
  model_equality) data_folder="model_equality" ;;
  unlearning)     data_folder="unlearning" ;;
  *) echo "unknown task: $task (watermark | model_equality | unlearning)" >&2; exit 1 ;;
esac

repo="$SLURM_SUBMIT_DIR"
data="${TEAM_DIR:?set TEAM_DIR in ~/.bashrc}/$data_folder"

source "$data/.venv/bin/activate"
export PYTHONPATH="$repo/$task:${PYTHONPATH:-}"
export WMLOC_MODELS="$data/models"
cd "$data"

echo "task=$task data=$data host=$(hostname) start=$(date -Is)"
srun python "$repo/$@"
echo "done $(date -Is)"
