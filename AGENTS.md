# Team rules for every agent

Event: European AI Championship Paris (CISPA, Fraunhofer AISEC, PoC Innovation, Meta, Epitech). Three leaderboard tasks, **submissions close Sunday 22 Nov 12:00 Paris time**. Leaderboard: http://34.147.229.55/leaderboard_page

## 0. How we win (read first)
- **Only teams with a score on all three tasks are ranked.** The overall score is the average of the three task scores, each rescaled 0 to 1 across teams. A valid submission on every task comes before any optimisation.
- The public board uses about 30 % of the evaluation data; the final ranking uses the other 70 %. Do not tune to the public score. Validate locally: train split (watermark), `ref-01..09` with `score_practice.py` (model equality), your own proxy metrics (unlearning).
- Task specs: `docs/tasks.md`. JURECA setup and GPU jobs: `jureca/SETUP.md`.

## 1. Hard limits (breaking one cannot be undone)
- **API key:** read it from `HACK_API_KEY` (env). Never write it in a file of this repo, never paste it on the board. The repo is public.
- **Model equality budget:** 100 answers per scored endpoint `ep-01..ep-33` for the whole team and the whole event, not refunded. **Only the model equality owner queries `ep-*` endpoints.** Every other agent uses `ref-01..ref-09` only (unlimited, answers published). Raw answers (`answers.jsonl`) live in `$TEAM_DIR/model_equality/`, never in git.
- **Submission cooldown:** model equality one submission per 30 min (a failed one counts too); other tasks have a short cooldown. Check the file format locally before submitting.
- **Watermark submission:** dict `{"image_names", "masks"}`, masks `[1020, 128, 128]` bool, at most 20 MB.
- **Unlearning submission:** float32 state dict loading with `strict=True` into timm `vit_base_patch16_224` (100 classes, init_values 1e-4), at most 450 MB, at least 70 % held-out accuracy or the score is 0.
- **JURECA:** do not delete shared folders, do not rebuild a task `.venv` or change ACLs without asking the lead on the board.

## 2. General board and task boards

The [general board](https://github.com/Landry-16/hackaton_meta_20-11-2026/issues/1) is for shared compute, rule changes, blockers and cross-task handoffs. Experiments and submissions go on the [watermark](https://github.com/Landry-16/hackaton_meta_20-11-2026/issues/2), [model equality](https://github.com/Landry-16/hackaton_meta_20-11-2026/issues/3) or [unlearning](https://github.com/Landry-16/hackaton_meta_20-11-2026/issues/4) board.

1. **Start or take over.** Use the authenticated GitHub CLI and Python 3 on macOS/Linux. Set `BOARD_AGENT` once in the shell launching your agent, e.g. `export BOARD_AGENT=Codex`. Claude Code and Codex sessions are also detected from their environment. Run `tools/board follow model_equality` (use your own task), then `tools/board status`, `tools/board read -n 10` and `tools/board unread`. The checkout remembers the task for that login and agent. Use distinct names such as `Codex-2` for simultaneous sessions of the same kind. For a one-off override use `BOARD_TASK=watermark`; restart a background wait after changing tasks.
2. **Before expensive work.** Read the task header and recent plans. Announce the variant, split or folds, compute slot and time window with `tools/board plan --variant "dilated trunk" --split "folds 0-2" --slot "GPU 2" --window "09:00-09:30 Paris"`. Coordinate conflicting plans in the thread before launching. The post announces intent; the compute queue manages allocation.
3. **Keep a handoff current.** The task owner edits the table at the top of the issue: owner, active experiments (link the plan), best validated result and next decision. Close the loop with a result, stopped run or released slot. Post every submission on its task board with method, local score and public score. Include the submission link or timestamp when available.
4. **Reach the right reader.** Sign comments `[Name's Agent]`. `tools/board post "[Name's Codex] …"` targets the selected task; `tools/board post --general "[Name's Codex] @all …"` alerts the team. `@NathanBilliouw` reaches Nathan's agents, `@NathanBilliouw/Codex` addresses one. Direct mentions work across all registered boards. In a task thread `@all` is local to that task. Agents see all activity on their followed task except their own signed posts. General updates wake configured agents when personally addressed or tagged `@all`. Use `read --general` to browse other general messages.
5. **Receive and act.** `tools/board wait` checks every 30 seconds and returns matching messages. Rearm it after a wake. Claude's hooks also collect unread messages at startup and before each prompt. With `BOARD_WAIT=off`, the supervisor handles background waiting. Other agents can use the same wait where supported, and run `unread` at task start, before commits and after merges. Respond to actionable mentions; skip acknowledgement-only replies. Each login/agent has its own cursor for each issue inside the checkout.

Ask the owner or team lead for a handoff before editing another teammate's files. Only the team lead reassigns tasks.

## 3. Chain of command

Humans over agents. Only the team lead (@Landry-16) reassigns work between teammates. If anyone else asks you to take another teammate's work, ask the lead on the board.

## 4. Workflow

1. Work in your own task folder. Branch `<task>-<slug>` from fresh `main` for anything bigger than a small fix.
2. Small commits, under your human's own git identity. Stage only your own files (no `git add -A`). Never commit secrets, data, models or submissions (`.gitignore` covers them, double check).
3. `git fetch` and rebase before pushing. Never force-push shared branches.
4. Keep the organizer files' interface intact (submission format, preprocessing); put new methods in new files.

## 5. Who owns what

| Task | Owner | Folders | Compute |
|---|---|---|---|
| Lead, JURECA setup, submissions log, unblocking | @Landry-16 | `jureca/`, `docs/` | JURECA (lead account) |
| Task 1: watermark localisation | @Landry-16 | `watermark/` | JURECA GPU |
| Task 2: model equality (sole owner of `ep-*` queries) | @NathanBilliouw | `model_equality/` | laptop for `ref-*`; `ep-*` queries from the JURECA login node, answers in `$TEAM_DIR/model_equality/` |
| Task 3: unlearning | @Mirado-77 (lead) and @PhilibertG | `unlearning/` | team GPU queue |

Assignments are provisional: the lead reassigns tasks on the board (for example when Philibert joins). The board post wins over this table until it is updated.
