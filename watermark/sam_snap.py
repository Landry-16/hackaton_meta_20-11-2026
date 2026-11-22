"""Snap a probability map onto object segments (sam_segments.py). Rules declared in LOOP.md before any result.
  greedy   moves = set one whole segment to 1 or to 0; apply the move that raises the expected score (soft mean IoU
           of the two classes against the probability map itself) most, until none raises it by 1e-4 (max 30 moves)
  vote     segments from the largest to the smallest, each set to the majority of its mean probability
Gate on the train OOF predictions (login node, no GPU):
  python sam_snap.py --source "pixelseal=synthv2400;trustmark=synthv3,synthv3b2000_long;stegastamp=real_synthv3300"
The oracle (best union of segments against the true mask) is diagnostic only: it selects nothing."""
import argparse

import numpy as np

from wmloc.utils import miou_image


def soft_score(cands, p):
    """Expected score of each candidate mask (rows, flat 0/1 float) against the probability map p (flat)."""
    q, n = 1 - p, 1 - cands
    i1, i0 = cands @ p, n @ q
    u1 = p.sum() + cands.sum(1) - i1
    u0 = q.sum() + n.sum(1) - i0
    return 0.5 * ((i1 + 1e-3) / (u1 + 1e-3) + (i0 + 1e-3) / (u0 + 1e-3))


def greedy(p, segs, start=None, target=None, moves=30, eps=1e-4):
    """p: 128 x 128 probabilities; segs: K x 128 x 128 bool. target: score against this map instead of p (oracle)."""
    t = (p if target is None else target).astype(np.float32).ravel()
    m = ((p >= 0.5) if start is None else start).astype(np.float32).ravel()
    if len(segs) == 0:
        return m.reshape(p.shape).astype(np.uint8)
    s = segs.reshape(len(segs), -1).astype(np.float32)
    best = soft_score(m[None], t)[0]
    for _ in range(moves):
        cands = np.concatenate([np.maximum(m[None], s), m[None] * (1 - s)])
        sc = soft_score(cands, t)
        k = int(np.argmax(sc))
        if sc[k] < best + eps:
            break
        m, best = cands[k], sc[k]
    return m.reshape(p.shape).astype(np.uint8)


def vote(p, segs):
    m = (p >= 0.5).astype(np.uint8)
    for s in segs:                                   # stored from the largest to the smallest
        m[s] = p[s].mean() >= 0.5
    return m


def main():
    from postproc import oof_probs
    from rect_search import METHODS
    from sam_segments import load_segments
    from wmloc.data import LocalisationDataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="method=oof source, separated by ;")
    a = ap.parse_args()
    src = dict(x.split("=") for x in a.source.split(";"))
    ds, segs = LocalisationDataset("release", "train"), load_segments("train")
    probs = {m: oof_probs(m, src[m]) for m in METHODS}
    rows = []
    for i in range(len(ds)):
        it = ds[i]
        name, method, full, mask = it["image_name"], it["method"], bool(it.get("full_watermark", False)), it["mask128"]
        if name not in segs or name not in probs[method]:
            continue
        p, sg = probs[method][name], segs[name]
        minority = (1 - mask) if full else mask
        ys, xs = np.nonzero(minority)
        rect = len(ys) > 0 and len(ys) / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) > 0.97
        kind = "A holes" if full else "B rect" if rect else "B object"
        const = np.full(mask.shape, int(full), np.uint8)
        outs = {"raw": (p >= 0.5).astype(np.uint8), "greedy": greedy(p, sg), "vote": vote(p, sg),
                "oracle": greedy(p, sg, start=const, target=mask, moves=60)}
        rows.append((method, kind, {k: miou_image(v, mask) for k, v in outs.items()}))
    print(f"{len(rows)} images with segments and OOF predictions")
    for method in list(METHODS) + ["all"]:
        for kind in ("A holes", "B object", "B rect", "all"):
            r = [x[2] for x in rows if method in (x[0], "all") and kind in (x[1], "all")]
            if r:
                print(f"{method:10s} {kind:9s} n={len(r):3d} | " + " | ".join(f"{k} {np.mean([x[k] for x in r]):.4f}" for k in ("raw", "greedy", "vote", "oracle")))
    for k in ("greedy", "vote"):
        d = np.array([x[2][k] - x[2]["raw"] for x in rows])
        print(f"{k} - raw: paired gain {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}")


if __name__ == "__main__":
    main()
