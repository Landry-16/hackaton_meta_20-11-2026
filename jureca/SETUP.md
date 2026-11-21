# JURECA setup

Project `training2666`, reservation `cispahack`, partition `dc-gpu`. Official guide: https://huggingface.co/datasets/SprintML/hackathon

## 0. Everyone, once (JuDoor)
1. Account on https://judoor.fz-juelich.de, MFA enabled, user agreement signed, project `training2666` visible.
2. Upload your SSH public key (`cat ~/.ssh/id_ed25519.pub`, or create it with `ssh-keygen -t ed25519`). It takes up to 15 min to work.
3. Log in: `ssh <juelich-user>@jureca.fz-juelich.de` (asks for your TOTP code).

## 1. Team lead only (Landry), once
```bash
mkdir -p ~/hackathon-scripts && cd ~/hackathon-scripts
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/hackathon_setup.sh -O hackathon_setup.sh
nano hackathon_setup.sh   # fill TEAM_LEAD, TEAMMATE_1..3 (Juelich usernames), TEAM_FOLDER
source hackathon_setup.sh
```
Then check that every task has its venv (the script skips a failed task and keeps going):
```bash
ls -d /p/scratch/training2666/$USER/<team>/*/.venv
```
Expected: `Hackathon_Watermark_Localisation_Task/.venv`, `model_equality/.venv`, `unlearning/.venv`. Post the result on the board.

## 2. Every other teammate, after the lead is done
```bash
mkdir -p ~/hackathon-scripts && cd ~/hackathon-scripts
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh -O teammate.sh
nano teammate.sh   # TEAM_LEAD = Landry's Juelich username, TEAM_FOLDER = same as lead
source teammate.sh
```
Never run both scripts from the same account. Do not rebuild venvs or re-download data.

## 3. Everyone: repo and env on JURECA
```bash
echo 'export TEAM_DIR=/p/scratch/training2666/<lead>/<team>' >> ~/.bashrc
echo 'export HACK_API_KEY=<team api key>' >> ~/.bashrc     # never commit it
source ~/.bashrc
mkdir -p $TEAM_DIR/code && cd $TEAM_DIR/code
git clone https://github.com/Landry-16/hackaton_meta_20-11-2026.git $USER && cd $USER
mkdir -p logs
```
Each person works in their own clone (`$TEAM_DIR/code/$USER`), the data stays in the task folders.

## 4. Every login
```bash
jutil env activate -p training2666
cd $TEAM_DIR/code/$USER && git pull
```

## 5. GPU jobs
```bash
sbatch -J unl-base jureca/job.sh unlearning unlearning/task_template.py
sbatch -J wm-base  jureca/job.sh watermark watermark/submission_template.py --data release --out submission.pt
squeue -u $USER
tail -f logs/unl-base-<jobid>.out
```
`jureca/job.sh` runs the script inside the task data folder with the task venv, so outputs (`submission.pt`...) land there.

Quick interactive GPU shell for debugging:
```bash
srun --account=training2666 --reservation=cispahack --partition=dc-gpu --gres=gpu:1 --cpus-per-task=8 --time=01:00:00 --pty bash
```

## Watermark, first run only
```bash
cd $TEAM_DIR/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
python $TEAM_DIR/code/$USER/watermark/download_models.py --models_dir $PWD/models
python -c "from wmloc.data import unzip; unzip('train.zip', 'release'); unzip('test.zip', 'release')"   # with PYTHONPATH=$TEAM_DIR/code/$USER/watermark
```
