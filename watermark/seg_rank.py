"""Learned choice of the mask of a type B image among candidates: every SAM segment (sam_segments.py), the
thresholded prediction, its largest component, the box of that component, and the two hand rules of sam_snap.py.
A small network scores each candidate from how well it agrees with the probability map (no image content, no
label), trained listwise on the train images to rank the candidates by their true score. Stacking on out-of-fold
probabilities: the ranker of a fold never sees the images it is judged on.
  python seg_rank.py --source "pixelseal=synthv3,synthv3b2000_long;trustmark=synthv3,synthv3b2000_long;stegastamp=real_synthv3300"
  add --apply to write output/rank_test.npz from the test probabilities (ranker trained on all train type B images)
Writes output/rank_train.npz (out-of-fold masks), read by postproc.py --seg as the option seg=rank."""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from sam_snap import greedy, vote
from wmloc.utils import miou_image

OUT = Path("output")
KINDS = ("segment", "raw", "lcc", "box", "greedy", "vote")
SEEDS, STEPS, TEMP = 5, 300, 0.05                # fixed before any result


def candidates(p, segs):
    raw = p >= 0.5
    out = [(s, "segment") for s in segs] + [(raw, "raw")]
    if raw.any():
        lab, n = ndimage.label(raw)
        lcc = lab == 1 + int(np.argmax(ndimage.sum(raw, lab, range(1, n + 1))))
        ys, xs = np.nonzero(lcc)
        box = np.zeros_like(raw); box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        out += [(lcc, "lcc"), (box, "box")]
    if len(segs):
        out += [(greedy(p, segs).astype(bool), "greedy"), (vote(p, segs).astype(bool), "vote")]
    return [(m, k) for m, k in out if m.any() and not m.all()]


def features(p, cands, method_id):
    m = np.stack([c[0] for c in cands]).reshape(len(cands), -1).astype(np.float32)
    pf, n = p.ravel().astype(np.float32), p.size
    area, sp = m.sum(1), pf.sum()
    inter = m @ pf
    soft1 = inter / (sp + area - inter)
    inter0 = (1 - m) @ (1 - pf)
    soft0 = inter0 / ((n - sp) + (n - area) - inter0)
    cols = [area / n, np.log(area / n), soft1, soft0, (soft1 + soft0) / 2, inter / area, (sp - inter) / (n - area), inter / max(sp, 1e-6)]
    for thr in (0.3, 0.5, 0.7):
        t = (pf >= thr).astype(np.float32)
        it = m @ t
        cols.append(it / np.maximum(area + t.sum() - it, 1))
    boxes = []
    for c, _ in cands:
        ys, xs = np.nonzero(c)
        boxes.append(c.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)))
    cols.append(np.array(boxes, np.float32))
    for v in (soft1, (soft1 + soft0) / 2):                      # position inside the image's own candidate list
        cols += [v - v.max(), np.argsort(np.argsort(-v)) / len(v)]
    cols += [np.full(len(cands), sp / n), np.full(len(cands), pf.max()), np.full(len(cands), len(cands) / 100)]
    kind = np.eye(len(KINDS), dtype=np.float32)[[KINDS.index(k) for _, k in cands]]
    meth = np.tile(np.eye(3, dtype=np.float32)[method_id], (len(cands), 1))
    return np.concatenate([np.stack(cols, 1).astype(np.float32), kind, meth], 1)


def fit(groups):
    """groups: list of (features C x F, true score of each candidate C). Returns a scoring function."""
    width = max(len(f) for f, _ in groups)
    x = torch.zeros(len(groups), width, groups[0][0].shape[1]); y = torch.zeros(len(groups), width); ok = torch.zeros(len(groups), width, dtype=torch.bool)
    for i, (f, s) in enumerate(groups):
        x[i, :len(f)], y[i, :len(f)], ok[i, :len(f)] = torch.from_numpy(f), torch.from_numpy(s), True
    mu, sd = x[ok].mean(0), x[ok].std(0) + 1e-6
    target = torch.softmax((y / TEMP).masked_fill(~ok, -1e9), 1)
    nets = []
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(x.shape[-1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        for _ in range(STEPS):
            logit = net((x - mu) / sd)[..., 0].masked_fill(~ok, -1e9)
            loss = -(target * torch.log_softmax(logit, 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        nets.append(net.eval())
    @torch.no_grad()
    def score(f):
        z = (torch.from_numpy(f) - mu) / sd
        return torch.stack([n(z)[:, 0] for n in nets]).mean(0).numpy()
    return score


def main():
    from postproc import context, oof_probs
    from rect_search import METHODS, folds_of
    from sam_segments import load_segments
    from wmloc.data import LocalisationDataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--test-prefix", default="seg_probs_ens")
    a = ap.parse_args()
    src = dict(x.split("=") for x in a.source.split(";"))
    ds, segs, info = LocalisationDataset("release", "train"), load_segments("train"), context("train")
    meta = {m["image_name"]: m for m in ds.meta}
    items = []                                                  # one per type B train image (type from the decode, as at test)
    for mi, method in enumerate(METHODS):
        probs = oof_probs(method, src[method])
        names = sorted(n for n in probs if meta[n]["method"] == method)
        fold = folds_of(np.array([method] * len(names)), np.array([bool(meta[n]["full_watermark"]) for n in names]))
        for n, k in zip(names, fold):
            if not info[n][0] or n not in segs:
                continue
            cands = candidates(probs[n], segs[n])
            gt = ds.m128[n]
            items.append({"name": n, "method": method, "fold": int(k), "cands": cands, "x": features(probs[n], cands, mi),
                          "y": np.array([miou_image(c.astype(np.uint8), gt) for c, _ in cands], np.float32),
                          "full": bool(meta[n]["full_watermark"]), "empty": miou_image(np.zeros_like(gt), gt)})
    print(f"{len(items)} type B train images, {np.mean([len(i['cands']) for i in items]):.0f} candidates per image", flush=True)
    picked = {}
    for k in range(5):
        score = fit([(i["x"], i["y"]) for i in items if i["fold"] != k])
        for i in items:
            if i["fold"] == k:
                i["pick"] = int(np.argmax(score(i["x"])))
                picked[i["name"]] = i["cands"][i["pick"]][0]
    def of(i, kind):                                            # score of the candidate of that kind (empty mask if it has none)
        j = [c for c, (_, kk) in enumerate(i["cands"]) if kk == kind]
        return i["y"][j[0]] if j else i["empty"]
    for method in list(METHODS) + ["all"]:
        for label, r in (("object+rect", [i for i in items if method in (i["method"], "all") and not i["full"]]),):
            d = np.array([i["y"][i["pick"]] - of(i, "vote") for i in r])
            print(f"{method:10s} n={len(r):3d} | " + " | ".join(f"{kk} {np.mean([of(i, kk) for i in r]):.4f}" for kk in ("raw", "greedy", "vote"))
                  + f" | ranker {np.mean([i['y'][i['pick']] for i in r]):.4f} | best candidate {np.mean([i['y'].max() for i in r]):.4f}"
                  + f" | ranker - vote {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f} | picks "
                  + " ".join(f"{kk}:{sum(i['cands'][i['pick']][1] == kk for i in r)}" for kk in KINDS), flush=True)
    np.savez(OUT / "rank_train.npz", names=np.array(list(picked)), masks=np.packbits(np.stack([picked[n].ravel() for n in picked]), axis=1))
    if a.apply:
        score = fit([(i["x"], i["y"]) for i in items])
        test, tsegs, tinfo = LocalisationDataset("release", "test"), load_segments("test"), context("test")
        out = {}
        for mi, method in enumerate(METHODS):
            z = np.load(OUT / f"{a.test_prefix}_{method}_test.npz")
            for n, p in zip(z["names"].tolist(), z["probs"].astype(np.float32)):
                if tinfo[n][0] and n in tsegs:
                    cands = candidates(p, tsegs[n])
                    out[n] = cands[int(np.argmax(score(features(p, cands, mi))))][0]
        np.savez(OUT / "rank_test.npz", names=np.array(list(out)), masks=np.packbits(np.stack([out[n].ravel() for n in out]), axis=1))
        print(f"wrote rank_test.npz: {len(out)} type B test images", flush=True)


if __name__ == "__main__":
    main()
