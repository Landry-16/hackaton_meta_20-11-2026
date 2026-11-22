"""Behavioural profile test: per response, features that LoRA moves (length, extra sentences, explanations, echo,
punctuation) and that sampling noise and watermarks move little. Pair statistic = sum over prompts of squared
standardized mean differences; p-value by permutation inside each prompt."""
import argparse, csv, itertools, json, re
from collections import defaultdict
import numpy as np
from mmd_equality import complete_linkage, practice_mcc

PROMPTS = {r["id"]: r["prompt"] for r in csv.DictReader(open("completion_prompts.csv", encoding="utf-8"))}


def feats(resp, prompt):
    r = resp.strip()
    tail = prompt.split(":", 1)[-1].strip()
    words = r.split()
    sents = [s for s in re.split(r"(?<=[.!?])\s+", r) if s]
    return [np.log1p(len(r)), np.log1p(len(words)), len(sents), r.count("\n") > 0, r.endswith("."),
            r.lower().startswith(tail[:20].lower()), '"' in r or "“" in r, bool(re.search(r"\d", r)),
            bool(re.search(r"\b(this|which|because|therefore|so|in this)\b", r.lower())), "`" in r]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("answers"); ap.add_argument("--max-per", type=int)
    ap.add_argument("--perm", type=int, default=300); ap.add_argument("--alpha", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    ap.add_argument("--verbose", action="store_true"); a = ap.parse_args()
    data = defaultdict(lambda: defaultdict(list))
    for l in open(a.answers, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r["id"] in PROMPTS and (a.max_per is None or len(data[r["id"]][r["endpoint"]]) < a.max_per):
                data[r["id"]][r["endpoint"]].append(feats(r["response"], PROMPTS[r["id"]]))
    eps = sorted({e for p in data for e in data[p]})
    F = {p: {e: np.array(v, float) for e, v in d.items()} for p, d in data.items()}
    scale = {p: np.vstack(list(d.values())).std(0) + 1e-3 for p, d in F.items()}
    rng = np.random.default_rng(0)
    def stat(p, A, B): return (((A.mean(0) - B.mean(0)) / scale[p]) ** 2).sum()
    tests = {}
    for x, y in itertools.combinations(eps, 2):
        ps = [p for p in F if x in F[p] and y in F[p]]
        obs = sum(stat(p, F[p][x], F[p][y]) for p in ps)
        hits = 0
        for _ in range(a.perm):
            s = 0
            for p in ps:
                pool = rng.permutation(np.vstack([F[p][x], F[p][y]])); n = len(F[p][x])
                s += stat(p, pool[:n], pool[n:])
            hits += s >= obs
        tests[(x, y)] = (obs, (hits + 1) / (a.perm + 1))
    if a.verbose:
        for k, (s, p) in sorted(tests.items(), key=lambda kv: kv[1][0]): print(f"  {k[0]} {k[1]} stat {s:7.2f} p {p:.3f}")
    for alpha in a.alpha:
        g = complete_linkage(eps, {k: v[1] for k, v in tests.items() if v[1] > alpha})
        print(f"alpha {alpha}: {[sorted(x) for x in g]}  MCC {practice_mcc(g)}")

if __name__ == "__main__":
    main()
