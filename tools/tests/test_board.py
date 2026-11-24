"""Offline regression tests for hackathon board routing. Run with python3 -m unittest discover -s tools/tests."""
import concurrent.futures
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from board_cli import Board, run


class BoardTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / ".github").mkdir()
        (self.root / ".github/boards.json").write_text(json.dumps(
            {"general": 1, "tasks": {"watermark": 2, "model_equality": 3, "unlearning": 4}}))
        self.rows = {i: [] for i in range(1, 5)}
        self.calls = []
        self.fail = set()
        self.env = patch.dict(os.environ, {"BOARD_AGENT": "Codex", "BOARD_TASK": "model_equality"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        api = patch.object(Board, "api", autospec=True, side_effect=self.api)
        api.start()
        self.addCleanup(api.stop)

    def api(self, board, suffix, *args):
        path = str(suffix).split("/")[0]
        issue = int(path)
        self.calls.append((suffix, args))
        if issue in self.fail:
            raise RuntimeError("unavailable")
        if "-f" in args:
            self.add(issue, "Nathan", args[args.index("-f") + 1][5:])
            return "comment-url"
        if "/comments?" in str(suffix):
            return "\n".join(json.dumps(row) for row in self.rows[issue])
        if args[-1] == ".comments":
            return str(len(self.rows[issue]))
        return "Task board\nOwner | Nathan"

    def add(self, issue, author, body):
        i = sum(map(len, self.rows.values())) + 1
        self.rows[issue].append(dict(id=i, created_at="2026-11-20T01:00:00Z", author=author, body=body))

    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run([str(self.root), "team/repo.with-dot", "Nathan", *args])
        return code, out.getvalue(), err.getvalue()

    def test_routing_and_two_agents_under_one_account(self):
        self.add(1, "Lead", "ordinary general chatter")
        self.add(1, "Lead", "@ALL team alert")
        self.add(3, "Nathan", "[Nathan's Codex] own signed post")
        self.add(3, "Nathan", "[Nathan's Claude] sibling result")
        self.add(3, "Nathan", "human note")
        self.add(4, "Lead", "@Nathan/Claude exclusive")
        self.add(4, "Lead", "@Nathan/Codex-2 distinct session")
        self.add(4, "Lead", "@NathanX different human")
        self.add(4, "Lead", "@Nathan human mention")
        self.add(4, "Lead", "@all local alert")
        self.add(2, "Lead", "@nathan/codex direct from watermark")
        _, text, _ = self.cli("unread")
        for included in ("team alert", "sibling result", "human note", "human mention", "direct from watermark"):
            self.assertIn(included, text)
        for excluded in ("chatter", "own signed", "exclusive", "distinct session", "different human", "local alert"):
            self.assertNotIn(excluded, text)
        self.assertEqual(self.cli("unread"), (0, "", ""))
        os.environ["BOARD_AGENT"] = "Claude"
        _, text, _ = self.cli("unread")
        self.assertIn("own signed post", text)
        self.assertIn("exclusive", text)
        self.assertNotIn("sibling result", text)
        self.assertNotIn("direct from watermark", text)

    def test_parallel_prompt_and_wait_consume_once(self):
        self.add(3, "Other", "one delivery")
        board = Board(self.root, "team/repo.with-dot", "Nathan")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            reads = list(pool.map(board.unread_issue, [3, 3]))
        self.assertEqual(sum(map(len, reads)), 1)
        self.assertEqual(board.unread_issue(3), [])

    def test_partial_failure_does_not_lose_or_replay_other_boards(self):
        self.add(2, "Other", "@Nathan/Codex recovered")
        self.add(3, "Other", "immediate")
        self.fail.add(2)
        code, text, err = self.cli("unread")
        self.assertEqual(code, 1)
        self.assertIn("immediate", text)
        self.assertIn("issue #2", err)
        self.fail.clear()
        code, text, _ = self.cli("unread")
        self.assertEqual(code, 0)
        self.assertNotIn("immediate", text)
        self.assertIn("recovered", text)

    def test_wait_ignores_chatter_and_wakes_when_addressed(self):
        self.add(1, "Lead", "background only")
        with patch("board_cli.time.sleep", side_effect=lambda _: self.add(2, "Lead", "@Nathan/Codex wake now")):
            code, text, _ = self.cli("wait", "--timeout", "1")
        self.assertEqual(code, 0)
        self.assertIn("wake now", text)
        self.assertNotIn("background only", text)
        with patch("board_cli.time.monotonic", side_effect=[10, 12]):
            self.assertEqual(self.cli("wait", "--timeout", "1")[0], 124)

    def test_first_poll_of_empty_task_then_burst(self):
        self.cli("unread")
        for i in range(21):
            self.add(3, "Other", f"activity {i}")
        text = self.cli("unread")[1]
        self.assertEqual(text.count("activity "), 21)
        self.assertTrue(any("--paginate" in args for _, args in self.calls))

    def test_plans_subscriptions_history_and_legacy_cursor(self):
        del os.environ["BOARD_TASK"]
        self.cli("follow", "watermark")
        self.cli("plan", "--variant", "dil", "--split", "0-2", "--slot", "GPU 2", "--window", "09:30")
        self.assertIn("RUN dil", self.rows[2][0]["body"])
        self.cli("post", "--general", "[Nathan's Codex] @all shared compute")
        self.assertIn("shared compute", self.rows[1][0]["body"])
        self.assertIn("RUN dil", self.cli("read")[1])
        self.assertIn("Owner", self.cli("status")[1])
        os.environ["BOARD_AGENT"] = "Claude"
        self.assertNotIn("RUN dil", self.cli("read")[1])
        os.environ.clear()
        self.add(1, "Other", "old post")
        (self.root / ".board").mkdir(exist_ok=True)
        (self.root / ".board/cursor-team_repo.with-dot-1").write_text("3 2026-11-20T01:00:00Z\n")
        self.add(1, "Other", "new legacy post")
        text = self.cli("unread")[1]
        self.assertNotIn("old post", text)
        self.assertIn("new legacy post", text)


if __name__ == "__main__":
    unittest.main()
