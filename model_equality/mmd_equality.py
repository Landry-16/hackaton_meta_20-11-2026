"""Pairwise model equality with a kernel two-sample test (MMD, Gao et al. 2025), many prompts x few samples.

For each pair of endpoints and each prompt, MMD^2 between their samples with a character n-gram kernel; the
statistic is the sum over prompts. A permutation test (labels shuffled inside each prompt) gives a p-value per pair,
so the decision threshold comes from the noise of the data itself, not from an absolute number. Groups are built by
complete linkage on the pairs that are not rejected.

  python mmd_equality.py completion_reference_answers.jsonl --practice
  python mmd_equality.py answers.jsonl --families pass2.csv --out submission.csv
"""
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent


def load(path, ids=None):
    data = defaultdict(lambda: defaultdict(list))
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if ids is None or r["id"] in ids:
                data[r["id"]][r["endpoint"]].append(r["response"].strip())
    return data


_EMB = {}


def embed_gram(texts, model):
    if model not in _EMB:
        from sentence_transformers import SentenceTransformer
        _EMB[model] = SentenceTransformer(model)
    E = _EMB[model].encode([t if t else " " for t in texts], normalize_embeddings=True, batch_size=64)
    return E @ E.T


def gram(texts, ngram=(1, 3), analyzer="char_wb"):
    if analyzer.startswith("embed:"):
        return embed_gram(texts, analyzer.split(":", 1)[1])
    if not any(t for t in texts):
        return np.ones((len(texts), len(texts)))
    X = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, sublinear_tf=True).fit_transform(
        [t if t else " " for t in texts])
    K = (X @ X.T).toarray()
    return K


def mmd2(K, ia, ib):
    """Unbiased MMD^2 between index sets ia and ib of the Gram matrix K."""
    na, nb = len(ia), len(ib)
    Kaa = K[np.ix_(ia, ia)]; Kbb = K[np.ix_(ib, ib)]; Kab = K[np.ix_(ia, ib)]
    return ((Kaa.sum() - np.trace(Kaa)) / (na * (na - 1)) + (Kbb.sum() - np.trace(Kbb)) / (nb * (nb - 1))
            - 2 * Kab.mean())


def pair_tests(data, eps, n_perm=300, seed=0, ngram=(1, 3), analyzer="char_wb"):
    """{(a, b): (statistic, p-value)} over all pairs, prompts pooled."""
    rng = np.random.default_rng(seed)
    prompts = sorted(data)
    grams = {}
    for p in prompts:
        texts, owner = [], []
        for e in eps:
            for t in data[p].get(e, []):
                texts.append(t); owner.append(e)
        grams[p] = (gram(texts, ngram, analyzer), np.array(owner))
    out = {}
    for a, b in itertools.combinations(eps, 2):
        parts = []
        for p in prompts:
            K, owner = grams[p]
            ia, ib = np.where(owner == a)[0], np.where(owner == b)[0]
            if len(ia) >= 2 and len(ib) >= 2:
                parts.append((K, ia, ib))
        if not parts:
            out[(a, b)] = (0.0, 1.0)
            continue
        obs = sum(mmd2(K, ia, ib) for K, ia, ib in parts)
        hits = 0
        for _ in range(n_perm):
            s = 0.0
            for K, ia, ib in parts:
                pool = rng.permutation(np.concatenate([ia, ib]))
                s += mmd2(K, pool[:len(ia)], pool[len(ia):])
            hits += s >= obs
        out[(a, b)] = (obs, (hits + 1) / (n_perm + 1))
    return out


def complete_linkage(eps, same_score):
    """Merge groups in order of evidence while every cross pair is 'same' (complete linkage)."""
    groups = [{e} for e in eps]
    for (a, b) in sorted(same_score, key=same_score.get, reverse=True):
        ga = next(g for g in groups if a in g); gb = next(g for g in groups if b in g)
        if ga is not gb and all((min(x, y), max(x, y)) in same_score for x in ga for y in gb):
            groups.remove(gb); ga |= gb
    return groups


def practice_mcc(groups, truth_path=ROOT / "practice_answers.csv"):
    truth = {r["endpoint"]: r["model"] for r in csv.DictReader(open(truth_path))}
    label = {e: i for i, g in enumerate(groups) for e in g}
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(sorted(truth), 2):
        t, g = truth[a] == truth[b], label[a] == label[b]
        tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
    d = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return (max(0.0, (tp * tn - fp * fn) / d) if d else 0.0), (tp, fp, fn, tn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers", nargs="+")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    ap.add_argument("--practice", action="store_true")
    ap.add_argument("--perm", type=int, default=300)
    ap.add_argument("--ngram", type=int, nargs=2, default=[1, 3])
    ap.add_argument("--analyzer", default="char_wb")
    ap.add_argument("--max-per", type=int, help="keep at most this many samples per prompt and endpoint")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    data = defaultdict(lambda: defaultdict(list))
    for path in args.answers:
        for p, d in load(path).items():
            for e, v in d.items():
                data[p][e].extend(v)
    if args.max_per:
        for p in data:
            for e in data[p]:
                data[p][e] = data[p][e][:args.max_per]
    eps = sorted({e for p in data for e in data[p]})
    tests = pair_tests(data, eps, args.perm, ngram=tuple(args.ngram), analyzer=args.analyzer)
    if args.verbose:
        for (a, b), (s, p) in sorted(tests.items(), key=lambda kv: kv[1][0]):
            print(f"  {a} {b}  stat {s:8.4f}  p {p:.3f}")
    for alpha in args.alpha:
        same = {k: v[1] for k, v in tests.items() if v[1] > alpha}
        groups = complete_linkage(eps, same)
        line = f"alpha {alpha}: groups {[sorted(g) for g in groups]}"
        if args.practice:
            mcc, cm = practice_mcc(groups)
            line += f"  practice MCC {mcc:.3f} {cm}"
        print(line)


if __name__ == "__main__":
    main()
