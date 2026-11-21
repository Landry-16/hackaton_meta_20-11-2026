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

## 2. The board (pinned GitHub issue)

Every agent reads and posts on the pinned "Message Board" issue via `tools/board` (needs only `gh`, logged in). Set `BOARD_ISSUE` if it is not issue #1.

- Start every post with `[<human>'s <agent>]`, e.g. `[Landry's Claude]`. Keep it short: what changed, which files or contracts, what you need, from whom.
- Mention by GitHub login: `@Landry-16` (Landry, team lead), `@NathanBilliouw` (Nathan), `@Mirado-77` (Mirado), `@PhilibertG` (Philibert), `@all` (everyone).
- **Post every submission** with task, method, local score and public score, so nobody wastes a cooldown on a worse idea.
- **Mentions you or `@all`:** act on it and reply promptly.
- **Not addressed to you:** still read it. Reassess your own work only if it is affected. Do not reply just to acknowledge.
- Before editing another teammate's files, ask for a handoff on the board and wait for the owner or the team lead.

Commands: `tools/board unread` (new posts, marks read; flags MENTION), `tools/board read -n 20`, `tools/board post "[Name's Agent] text"`, `tools/board wait` (blocks until someone else posts), `tools/board whoami`.

**Stay in the loop:**
- Claude Code: the committed hooks (`.claude/settings.json`) show unread posts at session start and on every prompt. Keep one background `tools/board wait` running and re-arm it after each wake.
- Everything else: run `tools/board unread` at task start, before each commit, and after merges.

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
| Task 1: watermark localisation | @NathanBilliouw | `watermark/` | JURECA GPU |
| Task 2: model equality (sole owner of `ep-*` queries) | @PhilibertG | `model_equality/` | laptop is enough (HTTP API) |
| Task 3: unlearning | @Mirado-77 | `unlearning/` | JURECA GPU |
