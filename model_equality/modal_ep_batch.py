"""Collect modal_probe prompts on the scored endpoints (review point 5b). Spends budget: dry run unless --go.

Rules built in (AGENTS.md section 1 and the review):
  - JURECA login node only; answers go to $TEAM_DIR/model_equality/modal_ep_answers.jsonl (never git).
  - Server counters must match answers.jsonl + this file before any request (no silent double spending).
  - Never more than 60 new answers per endpoint in total, and at least 10 left on every endpoint afterwards.
  - Round-robin over endpoints and repeats, so a run that stops early is still balanced.

  python modal_ep_batch.py --repeats 3            # dry run: prints the plan and the server counters
  python modal_ep_batch.py --repeats 3 --go       # batch 1
  python modal_ep_batch.py --repeats 7 --go       # batch 2 tops every prompt up to 7 repeats
"""
import argparse
import fcntl
import json
import os
import socket
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

import csv
from modal_probe import packed

URL = "http://34.147.229.55/33-model-equality/v1"
EPS = tuple(f"ep-{i:02d}" for i in range(1, 34))
TEAM = Path(os.environ.get("TEAM_DIR", "/p/scratch/training2666/savarimuthu1/teambuilders")) / "model_equality"
TRIO = TEAM / "answers.jsonl"
EXTRA = [TEAM / "modal_ep_answers.jsonl", TEAM / "completion_ep_answers.jsonl", TEAM / "reveal_ep_answers.jsonl", TEAM / "leak_ep_answers.jsonl", TEAM / "long_ep_answers.jsonl"]   # every budget-spending file
MAX_NEW = 60
RESERVE = 10


def counts(path):
    c, seen = Counter(), set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                k = (r["endpoint"], r["id"], r["repeat"])
                assert k not in seen, f"duplicate {k} in {path}"
                seen.add(k)
                c[r["endpoint"]] += 1
    return c, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, required=True, help="target repeats per prompt (cumulative)")
    ap.add_argument("--prompts", type=int, help="use only the first N packed prompts")
    ap.add_argument("--go", action="store_true", help="really query (spends budget)")
    ap.add_argument("--set", choices=["completion", "modal", "reveal", "leak", "long"], default="completion")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--eps", help="comma-separated subset of endpoints (default: all 33)")
    args = ap.parse_args()

    global OUT
    if args.set == "modal":
        prompts, OUT = packed(), EXTRA[0]
    else:
        here = Path(__file__).resolve().parent
        name = {"completion": "completion_prompts.csv", "reveal": "reveal_prompts.csv", "leak": "leak_prompts.csv", "long": "long_prompts.csv"}[args.set]
        prompts = [{"id": r["id"], "prompt": r["prompt"]}
                   for r in csv.DictReader(open(here / name, encoding="utf-8"))]
        OUT = {"completion": EXTRA[1], "reveal": EXTRA[2], "leak": EXTRA[3], "long": EXTRA[4]}[args.set]
    prompts = prompts[:args.prompts] if args.prompts else prompts
    trio, _ = counts(TRIO)
    mine, done = counts(OUT)
    spent = Counter()
    for f in EXTRA:
        spent.update(counts(f)[0])
    targets = args.eps.split(",") if args.eps else EPS
    assert all(t in EPS for t in targets)
    plan = [(r, p["id"], p["prompt"], ep) for r in range(args.repeats) for p in prompts for ep in targets
            if (ep, p["id"], r) not in done]
    new = Counter(ep for *_, ep in plan)
    for ep in EPS:
        if spent[ep] + new[ep] > MAX_NEW:
            raise SystemExit(f"{ep}: {spent[ep]} + {new[ep]} new answers would exceed the {MAX_NEW} cap")

    headers = {"X-API-Key": os.environ.get("HACK_API_KEY", "")}
    with requests.Session() as s:
        s.headers.update(headers)
        r = s.get(URL + "/models", timeout=30)
        r.raise_for_status()
        state = {m["id"]: m["budget"] for m in r.json()["data"] if m["id"] in EPS}
    bad = [ep for ep in EPS if state[ep]["used"] != trio[ep] + spent[ep]]
    if bad:
        raise SystemExit("Counters do not match the saved answers for " + ", ".join(
            f"{ep} (server {state[ep]['used']}, saved {trio[ep] + spent[ep]})" for ep in bad) + "; reconcile first.")
    low = [ep for ep in EPS if state[ep]["limit"] - state[ep]["used"] - new[ep] < RESERVE]
    if low:
        raise SystemExit(f"Would leave fewer than {RESERVE} answers on {', '.join(low)}")

    print(f"{len(prompts)} prompts x {args.repeats} repeats: {len(plan)} new answers "
          f"({max(new.values(), default=0)} per endpoint), already {sum(mine.values())} in {OUT}")
    print("Left after this batch:", sorted(Counter(state[ep]['limit'] - state[ep]['used'] - new[ep] for ep in EPS).items()))
    if not args.go:
        print("Dry run. Add --go to spend.")
        return
    if not socket.gethostname().startswith("jrlogin"):
        raise SystemExit("Run --go on the JURECA login node")
    if not headers["X-API-Key"]:
        raise SystemExit("HACK_API_KEY is not set")

    lock_file = (TEAM / "modal_collection.lock").open("a")   # pre-created: no new inode needed
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    used = {ep: state[ep]["used"] for ep in EPS}
    lock, stop = threading.Lock(), threading.Event()
    with OUT.open("a", buffering=1) as out, requests.Session() as s:
        s.headers.update(headers)

        def one(job):
            rep, pid, prompt, ep = job
            while not stop.is_set():
                reply = s.post(URL + "/chat/completions", timeout=(15, 300),
                               json={"model": ep, "max_tokens": args.max_tokens, "messages": [{"role": "user", "content": prompt}]})
                if reply.status_code == 429 and "Retry-After" in reply.headers:
                    stop.wait(max(1, int(reply.headers["Retry-After"])))
                    continue
                if reply.status_code != 200:
                    stop.set()
                    raise RuntimeError(f"{ep} {pid}: HTTP {reply.status_code} {reply.text[:200]}")
                body = reply.json()
                with lock:
                    out.write(json.dumps({"endpoint": ep, "id": pid, "repeat": rep, "prompt": prompt,
                                          "response": body["choices"][0]["message"]["content"]},
                                         ensure_ascii=False) + "\n")
                    os.fsync(out.fileno())
                    used[ep] += 1
                    if used[ep] != body["budget"]["used"]:
                        stop.set()
                        raise RuntimeError(f"{ep}: server counter {body['budget']['used']} != expected {used[ep]}")
                return

        with ThreadPoolExecutor(max_workers=4) as pool:
            for f in [pool.submit(one, j) for j in plan]:
                f.result()
    print("Done.", sorted(Counter(100 - used[ep] for ep in EPS).items()), "(answers left: endpoints)")


if __name__ == "__main__":
    main()
