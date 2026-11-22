"""Auto-submit a queue of checkpoints, one per server cooldown, and log every score to a CSV.

The server is our only judge (no gold model, no control set locally), so the goal is to never leave a cooldown slot
unused. Run it on the LOGIN node (compute nodes have no internet), detached so it survives logout:
  nohup $TEAM_DIR/unlearning/.venv/bin/python unlearning/submit_queue.py > logs/submit_queue.out 2>&1 &
Add checkpoints at any time, one path per line, in the order to submit (lines starting with # are ignored):
  echo $TEAM_DIR/unlearning/out/distill_..._e25.pt >> logs/queue.txt
Results: logs/submissions.csv (time, file, status, score, previous_best, message). Already-submitted files are skipped.
Stop it with: pkill -f submit_queue.py
"""
import argparse
import csv
import os
import re
import socket
import uuid
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

SERVER = "http://34.147.229.55"
TASK_ID = "31-unlearning"


def submit(path: Path, key: str) -> tuple:
    """Returns (status, score, previous_best, message, wait_seconds)."""
    with path.open("rb") as f:
        try:
            r = requests.post(f"{SERVER}/submit/{TASK_ID}", headers={"X-API-Key": key},
                              files={"file": (path.name, f)}, timeout=(10, 900))
        except requests.RequestException as exc:
            return "network_error", "", "", str(exc)[:200], 60
    try:
        body = r.json()
    except ValueError:
        body = {"detail": r.text[:300]}
    msg = str(body.get("message") or body.get("detail") or body)[:300]
    if r.status_code == 200 and body.get("status") == "success":
        return "success", body.get("score", ""), body.get("previous_best", ""), msg, 0
    m = re.search(r"(\d+)\s*(?:s\b|sec)", msg)
    if m:  # cooldown: not consumed, retry after the announced delay
        return "cooldown", "", "", msg, int(m.group(1)) + 5
    return f"rejected_{r.status_code}", "", "", msg, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queue", default="logs/queue.txt")
    p.add_argument("--log", default="logs/submissions.csv")
    p.add_argument("--cooldown", type=int, default=365, help="seconds between two submissions")
    p.add_argument("--poll", type=int, default=30, help="seconds between queue checks when it is empty")
    p.add_argument("--min-bytes", type=int, default=100_000_000, help="skip smaller files (empty reserved files)")
    p.add_argument("--lock", default="logs/submit_queue.lock",
                   help="single-instance lock: starting a new submitter makes any older one (any login node) exit")
    a = p.parse_args()
    key = os.environ.get("HACK_API_KEY", "")
    if not key:
        sys.exit("export HACK_API_KEY first (it is in ~/.bashrc on JURECA)")
    queue, log = Path(a.queue), Path(a.log)
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.touch()
    new_log = not log.exists()
    with log.open("a", newline="") as f:
        if new_log:
            csv.writer(f).writerow(["time", "file", "status", "score", "previous_best", "message"])

    lock, token = Path(a.lock), f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    lock.write_text(token)
    print(f"submitter {token} started (queue {queue}, log {log})", flush=True)

    next_slot = 0.0
    while True:
        if lock.read_text().strip() != token:
            print(f"{datetime.now():%H:%M:%S} another submitter took the lock, exiting", flush=True)
            return
        done = set()
        with log.open() as f:
            done = {row["file"] for row in csv.DictReader(f) if row["status"] == "success"
                    or row["status"].startswith("rejected")}
        todo = [line.strip() for line in queue.read_text().splitlines()
                if line.strip() and not line.startswith("#") and line.strip() not in done]
        if not todo:
            time.sleep(a.poll)
            continue
        path = Path(todo[0])
        if not path.is_file():
            print(f"{datetime.now():%H:%M:%S} missing {path}, skipped", flush=True)
            with log.open("a", newline="") as f:
                csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), str(path), "rejected_missing",
                                        "", "", "file not found"])
            continue
        time.sleep(max(0.0, next_slot - time.time()))
        if lock.read_text().strip() != token:
            print(f"{datetime.now():%H:%M:%S} another submitter took the lock, exiting", flush=True)
            return
        if not path.is_file():  # removed while we were waiting for the cooldown: skip it, never crash
            print(f"{datetime.now():%H:%M:%S} {path.name} disappeared before its slot, skipped", flush=True)
            with log.open("a", newline="") as f:
                csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), str(path), "rejected_missing",
                                        "", "", "file removed before submission"])
            continue
        if path.stat().st_size < a.min_bytes:  # reserved-but-never-written checkpoint (crashed run): never send it
            print(f"{datetime.now():%H:%M:%S} {path.name} is only {path.stat().st_size} bytes, skipped", flush=True)
            with log.open("a", newline="") as f:
                csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), str(path), "rejected_small",
                                        "", "", f"{path.stat().st_size} bytes"])
            continue
        print(f"{datetime.now():%H:%M:%S} submitting {path.name} ({len(todo) - 1} more queued)", flush=True)
        try:
            status, score, prev, msg, wait = submit(path, key)
        except OSError as exc:  # unreadable file etc.: log it, keep the submitter alive
            status, score, prev, msg, wait = "rejected_oserror", "", "", str(exc)[:200], 0
        print(f"  -> {status} score={score} prev_best={prev} {msg}", flush=True)
        if status in ("cooldown", "network_error"):
            next_slot = time.time() + wait
            continue
        with log.open("a", newline="") as f:
            csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), str(path), status, score, prev, msg])
        next_slot = time.time() + a.cooldown


if __name__ == "__main__":
    main()
