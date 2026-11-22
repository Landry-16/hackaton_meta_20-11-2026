"""Within-family model identity, disguise-aware (Claude Code rewrite, 22 Nov).

Facts measured on the practice endpoints (completion prompts, ref-01..09):
  - A watermark makes an endpoint near-deterministic: highest self-similarity, fewest distinct answers.
  - Temperature 1.0 makes it the most dispersed: lowest self-similarity.
  - Both are detected as 'different' by any two-sample test although they are the same model (7 of the 10 same
    pairs on the practice set), so they are recognised by their dispersion and attached to the base group.
  - System prompt and persona stay close to the plain endpoint; LoRA 0.7 and 1.5 are separated by an MMD test on
    sentence embeddings; fp8 and LoRA 0.3 are not separable at this budget (expected, Gao et al.).

Algorithm per family:
  1. Embed every answer (multilingual MiniLM), per prompt.
  2. Dispersion per endpoint: mean within-prompt cosine (self_sim) and share of distinct answers. Robust z-scores
     inside the family flag decoding disguises: watermark-like (low distinct share) and temperature-like (low
     self_sim).
  3. Pairwise MMD^2 (embedding kernel) summed over prompts, permutation p-value; complete linkage at alpha on the
     endpoints that are not flagged.
  4. Flagged endpoints join the largest group (the base model carries the disguises on the practice set).

  python family_classify.py completion_reference_answers.jsonl --practice
  python family_classify.py completion_ep_answers.jsonl --families pass2.csv --out submission.csv
"""
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mmd_equality import complete_linkage, mmd2, practice_mcc

ROOT = Path(__file__).resolve().parent
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_M = None
EXACT_W = 0.0   # weight of the exact-match kernel on normalised answer prefixes (shared hallucinations)


def key(text, words=6):
    import re
    return " ".join(re.findall(r"[\w']+", text.lower())[:words])


def encoder():
    global _M
    if _M is None:
        from sentence_transformers import SentenceTransformer
        _M = SentenceTransformer(MODEL)
    return _M


def load(paths, ids=None):
    data = defaultdict(lambda: defaultdict(list))
    for path in paths:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if ids is None or r["id"] in ids:
                    data[r["id"]][r["endpoint"]].append(r["response"].strip())
    return data


def robust_z(values):
    v = np.array(list(values.values()), float)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
    return {k: (x - med) / mad for k, x in values.items()}


def set_test(member, group, prompts, grams, owners, n_perm, rng):
    """Permutation p-value of `member` against the pooled answers of `group` (more power than pairwise tests)."""
    parts = []
    for p in prompts:
        ia = np.where(owners[p] == member)[0]
        ib = np.where(np.isin(owners[p], list(group)))[0]
        parts.append((grams[p], ia, ib))
    obs = sum(mmd2(K, ia, ib) for K, ia, ib in parts)
    hits = 0
    for _ in range(n_perm):
        s = 0.0
        for K, ia, ib in parts:
            pool = rng.permutation(np.concatenate([ia, ib]))
            s += mmd2(K, pool[:len(ia)], pool[len(ia):])
        hits += s >= obs
    return (hits + 1) / (n_perm + 1)


def grow_base(core, prompts, grams, owners, tests, alpha, n_perm, rng):
    """Seed the base group with the closest pair, then add the endpoint that the pooled group rejects least, while
    the test does not reject at alpha. Endpoints left over are grouped among themselves by complete linkage."""
    seed_pair = min(tests, key=lambda k: tests[k][0])
    base, rest = set(seed_pair), [e for e in core if e not in seed_pair]
    if tests[seed_pair][1] <= alpha:
        base, rest = set(), list(core)
    while base and rest:
        pv = {e: set_test(e, base, prompts, grams, owners, n_perm, rng) for e in rest}
        best = max(pv, key=pv.get)
        if pv[best] <= alpha:
            break
        base.add(best); rest.remove(best)
    others = complete_linkage(rest, {k: v[1] for k, v in tests.items() if v[1] > alpha and k[0] in rest
                                     and k[1] in rest})
    return ([base] if base else []) + others


def classify(data, eps, alpha=0.05, z_cut=2.5, n_perm=300, seed=0, verbose=False, grow=True):
    rng = np.random.default_rng(seed)
    prompts = [p for p in sorted(data) if all(len(data[p].get(e, [])) >= 2 for e in eps)]
    grams, owners, self_sim, distinct = {}, {}, defaultdict(list), defaultdict(list)
    for p in prompts:
        texts, owner = [], []
        for e in eps:
            texts += data[p][e]; owner += [e] * len(data[p][e])
        E = encoder().encode([t or " " for t in texts], normalize_embeddings=True, batch_size=64)
        K = E @ E.T
        if EXACT_W:
            keys = [key(t) for t in texts]
            X = np.array([[a == b for b in keys] for a in keys], float)
            K = (1 - EXACT_W) * K + EXACT_W * X
        owner = np.array(owner)
        grams[p], owners[p] = K, owner
        for e in eps:
            idx = np.where(owner == e)[0]
            sub = K[np.ix_(idx, idx)]
            n = len(idx)
            self_sim[e].append((sub.sum() - np.trace(sub)) / (n * (n - 1)))
            distinct[e].append(len(set(data[p][e])) / n)
    ss = {e: float(np.mean(self_sim[e])) for e in eps}
    dist = {e: float(np.mean(distinct[e])) for e in eps}
    z_ss, z_dist = robust_z(ss), robust_z(dist)
    flags = {}
    for e in eps:
        if z_dist[e] <= -z_cut or z_ss[e] >= z_cut:
            flags[e] = "watermark-like"
        elif z_ss[e] <= -z_cut:
            flags[e] = "temperature-like"
    core = [e for e in eps if e not in flags]
    tests = {}
    for a, b in itertools.combinations(core, 2):
        parts = []
        for p in prompts:
            ia, ib = np.where(owners[p] == a)[0], np.where(owners[p] == b)[0]
            parts.append((grams[p], ia, ib))
        obs = sum(mmd2(K, ia, ib) for K, ia, ib in parts)
        hits = 0
        for _ in range(n_perm):
            s = 0.0
            for K, ia, ib in parts:
                pool = rng.permutation(np.concatenate([ia, ib]))
                s += mmd2(K, pool[:len(ia)], pool[len(ia):])
            hits += s >= obs
        tests[(a, b)] = (obs, (hits + 1) / (n_perm + 1))
    if grow and len(core) >= 2:
        groups = grow_base(core, prompts, grams, owners, tests, alpha, n_perm, rng)
    else:
        groups = complete_linkage(core, {k: v[1] for k, v in tests.items() if v[1] > alpha})
    groups.sort(key=len, reverse=True)
    if groups:
        for e in flags:
            groups[0].add(e)
    else:
        groups = [set(flags)]
    if verbose:
        for e in eps:
            print(f"  {e} self_sim {ss[e]:.3f} (z {z_ss[e]:+.1f})  distinct {dist[e]:.2f} (z {z_dist[e]:+.1f})"
                  f"  {flags.get(e, '')}")
        for (a, b), (s, p) in sorted(tests.items(), key=lambda kv: kv[1][0]):
            print(f"  {a} {b}  mmd {s:7.3f}  p {p:.3f}")
    return groups, flags, tests, len(prompts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers", nargs="+")
    ap.add_argument("--families", help="csv endpoint,family,... (scored endpoints); refs = one family")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--z", type=float, default=2.5)
    ap.add_argument("--max-per", type=int)
    ap.add_argument("--seeds", type=int, default=1, help="random subsamples of --max-per answers to average over")
    ap.add_argument("--extra", nargs="*", default=[], help="more answer files (e.g. the 3 trio prompts)")
    ap.add_argument("--extra-ids", default="alice,gettysburg,code-import")
    ap.add_argument("--max-extra", type=int, default=10, help="cap per prompt and endpoint for --extra answers")
    ap.add_argument("--practice", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--exact", type=float, default=0.0, help="weight of the exact-prefix kernel")
    args = ap.parse_args()
    global EXACT_W
    EXACT_W = args.exact
    full = load(args.answers)
    fam = {}
    if args.families:
        fam = {r["endpoint"]: r["family"] for r in csv.DictReader(open(args.families))}
    scores, rows = [], []
    for seed in range(args.seeds):
        data = full
        if args.max_per:
            rng = np.random.default_rng(seed)
            data = {p: {e: [v[i] for i in sorted(rng.choice(len(v), min(args.max_per, len(v)), replace=False))]
                        for e, v in d.items()} for p, d in full.items()}
        if args.extra:
            ids = set(args.extra_ids.split(","))
            ext = load(args.extra, ids)
            rng = np.random.default_rng(1000 + seed)
            data = {p: dict(d) for p, d in data.items()}
            for p, d in ext.items():
                key = p + "#trio"   # kept apart from a completion prompt with the same id (different max_tokens)
                data[key] = {e: [v[i] for i in sorted(rng.choice(len(v), min(args.max_extra, len(v)), replace=False))]
                             for e, v in d.items()}
        eps_all = sorted({e for p in data for e in data[p]})
        families = defaultdict(list)
        for e in eps_all:
            families[fam.get(e, "P")].append(e)
        rows = []
        all_groups = []
        for F, eps in sorted(families.items()):
            groups, flags, tests, n_p = classify(data, eps, args.alpha, args.z, seed=seed, verbose=args.verbose)
            print(f"[seed {seed}] family {F} ({n_p} prompts): {[sorted(g) for g in groups]}  flags {flags}")
            all_groups += groups
            for i, g in enumerate(groups):
                for e in sorted(g):
                    rows.append((e, F, f"{F}-M{i}"))
        if args.practice:
            mcc, cm = practice_mcc(all_groups)
            scores.append(mcc)
            print(f"[seed {seed}] practice MCC {mcc:.3f} {cm}")
    if args.practice and len(scores) > 1:
        print(f"practice MCC over {len(scores)} subsamples: mean {np.mean(scores):.3f} min {np.min(scores):.3f}")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["endpoint", "family", "model"])
            for r in sorted(rows):
                w.writerow(r)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
