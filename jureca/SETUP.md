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
for u in <user2> <user3> <user4>; do id "$u" >/dev/null && echo "ok $u"; done   # a wrong name aborts the script at the ACL step
nano hackathon_setup.sh   # fill TEAM_LEAD, TEAMMATE_1..3 (Juelich usernames that printed ok), TEAM_FOLDER
source hackathon_setup.sh
```
Only teammates whose username already works can be listed; add late ones afterwards with section 6 of the official README.

The script only builds a venv when the dataset root has a `requirements.txt`. The watermark one is inside `starter_kit.zip`, so build it by hand (Python 3.11, as the kit is tested with torch 2.5.1):
```bash
cd /p/scratch/training2666/$USER/<team>/Hackathon_Watermark_Localisation_Task
unzip -o starter_kit.zip
uv venv -p 3.11 .venv && VIRTUAL_ENV=$PWD/.venv uv pip install -r requirements.txt
```
Then check the three venvs and post the result on the board:
```bash
ls -d /p/scratch/training2666/$USER/<team>/*/.venv
```
Expected: `Hackathon_Watermark_Localisation_Task/.venv`, `model_equality/.venv`, `unlearning/.venv`.

## 2. Every other teammate, after the lead is done
```bash
mkdir -p ~/hackathon-scripts && cd ~/hackathon-scripts
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh -O teammate.sh
sed -i -e "s/^TEAM_LEAD=\"\"/TEAM_LEAD=\"savarimuthu1\"/" -e "s/^TEAM_FOLDER=\"\"/TEAM_FOLDER=\"teambuilders\"/" teammate.sh
source teammate.sh
```
Never run both scripts from the same account. Do not rebuild venvs or re-download data.

## 3. Everyone: repo and env on JURECA
```bash
echo 'export TEAM_DIR=/p/scratch/training2666/savarimuthu1/teambuilders' >> ~/.bashrc
echo 'export HACK_API_KEY=<team api key>' >> ~/.bashrc     # never commit it
source ~/.bashrc
mkdir -p $TEAM_DIR/code && cd $TEAM_DIR/code
git clone git@github.com:Landry-16/hackaton_meta_20-11-2026.git $USER && cd $USER
mkdir -p logs
```
The repo is private: log in with agent forwarding (`ssh -A <user>@jureca.fz-juelich.de`, your GitHub SSH key loaded with `ssh-add`) so clone and push use your own GitHub key.
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

## Gotchas
- **GPU nodes have no internet.** Anything that downloads on first use (weights, pip, Hugging Face) must run once on the login node first, or the job dies with `Address family not supported by protocol`.
- Default job time is 1 h; `jureca/job.sh` asks for 4 h.
- **One node per person at a time** (pending jobs count). The organizers cancel extra jobs and revoke access after repeated offenses. `jureca/run.sh` refuses to queue a second job.

## Team GPU queue (one node per TEAM)
Organizers: at most ONE job for the whole team, running or queued. Only Landry's `hold` job exists; everyone runs on it through a shared queue:
```bash
cd $TEAM_DIR/code/$USER && git pull
bash jureca/gpu_submit.sh "source $TEAM_DIR/unlearning/.venv/bin/activate \&\& python my_script.py"   # runs in the current folder, on a free GPU
bash jureca/gpu_submit.sh status
bash jureca/gpu_submit.sh log <name>
```
The command runs on Landry's account, no internet (download on the login node first), outputs inside `$TEAM_DIR`. Never `sbatch` / `salloc` yourself.
