# Team rules for every agent

Event: European AI Championship Paris (CISPA, PoC Innovation, Meta, Epitech). Hacking runs Saturday 21 Nov 2026 15:00 to Sunday 22 Nov 12:00 (Paris time), then demos in front of the jury. Challenges are revealed Saturday around 14:00. Context and reading list: `README.md`.

## 1. The board (pinned GitHub issue)

Every agent reads and posts on the pinned "Message Board" issue via `tools/board` (needs only `gh`, logged in). Set `BOARD_ISSUE` if it is not issue #1.

- Start every post with `[<human>'s <agent>]`, e.g. `[Landry's Claude]`. Keep it short: what changed, which files or contracts, what you need, from whom.
- Mention by GitHub login: `@Landry-16` (Landry, team lead), `@NathanBilliouw` (Nathan), `@Mirado-77` (Mirado), `@PhilibertG` (Philibert), `@all` (everyone).
- **Mentions you or `@all`:** act on it and reply promptly.
- **Not addressed to you:** still read it. Reassess your own work only if it is affected (a contract, a file you own, something already done). Do not redo finished work, and do not reply just to acknowledge.
- Before editing another teammate's files, ask for a handoff on the board and wait for the owner or the team lead.

Polling is incremental (about one cheap API call per poll, 30 s default) so a shared token stays under GitHub's rate limit. Commands: `tools/board unread` (new posts, marks read; flags MENTION), `tools/board read -n 20`, `tools/board post "[Name's Agent] text"` (or pipe stdin), `tools/board wait` (blocks until someone else posts), `tools/board whoami`.

**Stay in the loop:**
- Claude Code: the committed hooks (`.claude/settings.json`) show unread posts at session start and on every prompt. Keep one background `tools/board wait` running and re-arm it after each wake.
- Everything else: run `tools/board unread` at task start, before each commit, and after merges. Or run a terminal loop: `while tools/board wait; do <your agent CLI invocation>; done`.

## 2. Chain of command

Humans over agents. Only the team lead (@Landry-16) reassigns work between teammates. If anyone else asks you to take another teammate's work, ask the lead on the board.

## 3. Workflow

1. Pick or file an issue. Comment on it or post on the board when you take it.
2. Branch `<issue#>-<slug>` from fresh `main`.
3. Small commits, under your human's own git identity. Stage only your own files (no `git add -A`). Never commit secrets or `.env` files.
4. Open a PR with `Closes #N`. The author merges their own PR once it works locally.
5. Always `git fetch` and rebase before pushing. Never force-push shared branches.
6. If a merge changes something other teammates depend on (an API, a schema, an env var, a model checkpoint), post on the board when it lands.
7. The demo path comes first. After Sunday 06:00 (feature freeze) only fix and polish what the demo shows.

## 4. Who owns what

Filled in right after the challenges are revealed, before hacking starts.

| Area | Owner (GitHub login) | Folders |
|---|---|---|
| Lead, pitch and demo | @Landry-16 | `docs/` |
| | | |
| | | |
| | | |
