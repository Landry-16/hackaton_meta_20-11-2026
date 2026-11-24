"""Per-agent subscriptions and issue history for tools/board (Python 3, POSIX)."""
import argparse
import concurrent.futures
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


SIGNATURE = re.compile(r"^\[[^\]\n]+'s ([^\]\n]+)\]")
MENTIONS = re.compile(r"(?<![\w@/-])@([\w-]+)(?:/([\w-]+))?(?![\w/-])", re.ASCII)
COMMENT = "{id, created_at, author: .user.login, body}"


def integer(value):
    value = int(value)
    if value < 1:
        raise ValueError("expected a positive number")
    return value


def output(*args):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"{args[0]} exited {result.returncode}")
    return result.stdout


def current_agent():
    if os.environ.get("BOARD_AGENT"):
        return os.environ["BOARD_AGENT"]
    if os.environ.get("CLAUDECODE") or os.environ.get("CLAUDE_PROJECT_DIR"):
        return "Claude"
    if os.environ.get("CODEX_THREAD_ID"):
        return "Codex"
    return ""


class Board:
    def __init__(self, root, repo, login, changing_task=False):
        self.root, self.repo, self.login = root, repo, login
        config = root / ".github/boards.json"
        boards = json.loads(config.read_text()) if config.exists() else {}
        self.general = integer(os.environ.get("BOARD_ISSUE") or boards.get("general", 1))
        self.tasks = {k: integer(v) for k, v in boards.get("tasks", {}).items()}
        self.agent = current_agent()
        if self.agent and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.agent):
            raise ValueError("BOARD_AGENT must be a name like Claude, Codex, or Codex-2")
        self.profile = f"board.{login.lower()}.{self.agent.lower() or 'default'}.task"
        stored = subprocess.run(["git", "-C", str(root), "config", "--get", self.profile],
                                stdout=subprocess.PIPE, text=True)
        selected = os.environ.get("BOARD_TASK") or stored.stdout.strip() or "general"
        self.task = self.general if changing_task else self.issue_for(selected)

    def issue_for(self, name):
        if name == "general":
            return self.general
        if name in self.tasks:
            return self.tasks[name]
        if name.isascii() and name.isdecimal():
            return integer(name)
        raise ValueError(f"unknown board {name!r}; choose general, an issue number, or {', '.join(self.tasks)}")

    def api(self, suffix, *args):
        return output("gh", "api", f"repos/{self.repo}/issues/{suffix}", *args)

    def comments(self, issue, query, paginate=False):
        flags = ["--paginate"] if paginate else []
        data = self.api(f"{issue}/comments?per_page=100&{query}", *flags, "--jq", ".[] | " + COMMENT)
        return [json.loads(line) for line in data.splitlines()]

    def last(self, issue, count):
        total = int(self.api(issue, "--jq", ".comments"))
        page = (total + 99) // 100
        result = []
        while page and len(result) < count:
            result = self.comments(issue, f"page={page}") + result
            page -= 1
        return result[-count:] if count else []

    def directed(self, body, issue):
        for handle, agent in MENTIONS.findall(body.casefold()):
            if handle == self.login.casefold() and (not agent or agent == self.agent.casefold()):
                return True
            if handle == "all" and not agent and issue in (self.general, self.task):
                return True
        return False

    def own(self, row):
        if row["author"].casefold() != self.login.casefold():
            return False
        signature = SIGNATURE.match(row["body"])
        return not self.agent or bool(signature and signature[1].casefold() == self.agent.casefold())

    def render(self, row, issue):
        flag = " ** MENTION **" if self.directed(row["body"], issue) else ""
        return f"--- issue #{issue} @{row['author']} {row['created_at']} (#{row['id']}){flag}\n{row['body']}"

    def unread_issue(self, issue):
        state = self.root / ".board"
        state.mkdir(exist_ok=True)
        # Keep the original cursor format and filename for single-board clients.
        key = f"cursor-{self.repo.replace('/', '_')}-{issue}"
        if self.agent or self.task != self.general:
            key += f".{self.login.lower()}.{self.agent.lower() or 'default'}"
        cursor = state / key
        # A prompt hook and a background wait can consume this same inbox concurrently.
        # The OS releases this per-reader lock even if a hook is killed.
        with (state / (key + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            saved = cursor.read_text().split() if cursor.exists() else []
            if len(saved) == 2:
                last_id = int(saved[0])
                rows = [r for r in self.comments(issue, "since=" + saved[1], paginate=True)
                        if r["id"] > last_id]
            else:
                rows = self.last(issue, 10)
            if rows:
                newest = max(rows, key=lambda r: r["id"])
                position = f"{newest['id']} {newest['created_at']}\n"
            elif len(saved) != 2:
                # An empty issue has no history to replay. Fetch all its new comments,
                # even if more than ten arrive between now and the next poll.
                position = "0 1970-01-01T00:00:00Z\n"
            else:
                return []
            single_board = self.task == self.general and (not self.agent or not self.tasks)
            followed_task = issue == self.task and issue != self.general
            visible = [self.render(r, issue) for r in rows if not self.own(r) and
                       (single_board or followed_task or self.directed(r["body"], issue))]
            tmp = state / (key + ".tmp")
            tmp.write_text(position)
            tmp.replace(cursor)
            return visible

    def poll(self):
        issues = [self.general]
        if self.agent or self.task != self.general:
            issues = list(dict.fromkeys([self.general, self.task, *self.tasks.values()]))
        messages, errors = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(issues))) as pool:
            jobs = [(i, pool.submit(self.unread_issue, i)) for i in issues]
            for issue, job in jobs:
                try:
                    messages.extend(job.result())
                except (RuntimeError, ValueError, OSError) as exc:
                    errors.append(f"issue #{issue}: {exc}")
        return messages, errors

    def post(self, issue, body):
        signature = SIGNATURE.match(body)
        if not signature:
            raise ValueError("start the post with [Name's Agent]")
        if self.agent and signature[1].casefold() != self.agent.casefold():
            raise ValueError(f"sign this session's post with {self.agent} or set BOARD_AGENT")
        print(self.api(f"{issue}/comments", "-f", f"body={body}", "--jq", ".html_url").strip())


def parser():
    cli = argparse.ArgumentParser(prog="tools/board", description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("read", "post", "status"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--general", action="store_true", help="use the general board")
        if name == "read":
            cmd.add_argument("-n", type=int, default=10)
        if name == "post":
            cmd.add_argument("text", nargs="*")
    commands.add_parser("unread")
    wait = commands.add_parser("wait")
    wait.add_argument("--interval", type=integer, default=30)
    wait.add_argument("--timeout", type=int, default=0)
    commands.add_parser("follow").add_argument("task")
    plan = commands.add_parser("plan", help="announce an expensive run on the selected task")
    for name in ("variant", "split", "slot", "window"):
        plan.add_argument("--" + name, required=True)
    commands.add_parser("help")
    return cli


def run(argv):
    root, repo, login, *args = argv
    cli = parser()
    if not args or args == ["help"]:
        cli.print_help()
        return 0
    args = cli.parse_args(args)
    board = Board(Path(root), repo, login, changing_task=args.command == "follow")
    issue = board.general if getattr(args, "general", False) else board.task
    if args.command == "follow":
        board.issue_for(args.task)
        output("git", "-C", root, "config", "--local", board.profile, args.task)
        print(f"{login}/{board.agent or 'default'} follows {args.task} (BOARD_TASK overrides this setting).")
    elif args.command == "read":
        if args.n < 0:
            raise ValueError("-n must be nonnegative")
        for row in board.last(issue, args.n):
            print(board.render(row, issue))
    elif args.command == "status":
        print(f"#{issue} " + board.api(issue, "--jq", r'.title + "\n" + .body').rstrip())
    elif args.command == "post":
        board.post(issue, " ".join(args.text) if args.text else sys.stdin.read())
    elif args.command == "plan":
        if board.task == board.general or not board.agent:
            raise ValueError("select a task with follow and an identity with BOARD_AGENT before posting a plan")
        board.post(issue, f"[{login}'s {board.agent}] RUN {args.variant} | split: {args.split} | "
                   f"compute: {args.slot} | window: {args.window}")
    else:
        waiting = args.command == "wait"
        if waiting and args.timeout < 0:
            raise ValueError("--timeout must be nonnegative")
        start = time.monotonic()
        while True:
            messages, errors = board.poll()
            for message in messages:
                print(message)
            for error in errors:
                print("board: " + error, file=sys.stderr)
            if not waiting or messages:
                return int(bool(errors) and not waiting)
            delay = args.interval
            if args.timeout:
                remaining = args.timeout - (time.monotonic() - start)
                if remaining <= 0:
                    print("board: wait timed out", file=sys.stderr)
                    return 124
                delay = min(delay, remaining)
            time.sleep(delay)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run(sys.argv[1:]))
    except (ValueError, OSError, RuntimeError) as error:
        sys.exit(f"board: {error}")
