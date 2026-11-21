"""Metric-aligned post-processing of segmenter probabilities, tuned by nested CV on out-of-fold predictions.
Per method, a few rules (chosen on 4 folds, scored on the 5th):
  thr      probability threshold
  lcc      type B (not decodable as a whole): keep only the largest connected component
  snap     type B: replace that component by its bounding box when it fills it at least `snap` (1.01 = never)
  floor    never return a constant mask: if empty (or full), take the `floor` share of cells with the highest
           (or lowest) probability, since every train image holds both classes
  rect     a rectangle whose crop decodes the message (rect_search) overrides the mask
  python postproc.py                      CV report and plots on the OOF files
  python postproc.py --apply              also writes output/submission_post.pt from the test probabilities
OOF files: output/oof_ext_<method>_fold<k>_synth400.npz (seg_ext) or output/oof_real_fold<k>_noflip.npz (seg_real)."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from rect_search import METHODS, box_mask, choose, folds_of, load_cache
from wmloc.data import LocalisationDataset
from wmloc.utils import miou_image

OUT = Path("output")
GRID = {"thr": [0.3, 0.4, 0.5, 0.6], "lcc": [False, True], "snap": [0.8, 0.9, 1.01], "floor": [0.0, 0.03, 0.08],
        "refine": [0.8, 0.9, 2.0],           # 2.0 = never use the refined rectangle
        "rect": [True, False]}               # use the decoded rectangle of rect_search or not


def oof_probs(method, source):
    probs = {}
    for k in range(5):
        # source: "synth400" / "synthv2400" (seg_ext) or "real<suffix>" (seg_real), e.g. real_noflip, real_synthv2300
        f = OUT / (f"oof_real_fold{k}{source[4:]}.npz" if source.startswith("real") else f"oof_ext_{method}_fold{k}_{source}.npz")
        if f.exists():
            z = np.load(f)
            probs.update(dict(zip(z["names"].tolist(), z["probs"].astype(np.float32))))
    return probs


def post(p, typeb, rect_box, prm, refined=None):
    if rect_box is not None and prm.get("rect", True):
        return box_mask(rect_box)
    if refined is not None and refined[0] >= prm.get("refine", 2.0):
        return box_mask(refined[1])
    m = (p >= prm["thr"]).astype(np.uint8)
    if typeb and prm["lcc"] and m.any():
        lab, n = ndimage.label(m)
        if n > 1:
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            m = (lab == 1 + int(np.argmax(sizes))).astype(np.uint8)
    if typeb and m.any() and prm["snap"] <= 1:
        ys, xs = np.nonzero(m)
        box = np.zeros_like(m)
        box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
        if m.sum() / box.sum() >= prm["snap"]:
            m = box
    if prm["floor"] > 0 and (m.all() or not m.any()):
        k = max(1, int(prm["floor"] * m.size))
        order = np.argsort(p.ravel())
        m = np.ones(m.size, np.uint8) if m.all() else np.zeros(m.size, np.uint8)
        m[order[:k] if m.all() else order[-k:]] = 0 if m.all() else 1
        m = m.reshape(p.shape)
    return m


def context(split):
    cache, params = load_cache(split), json.loads((OUT / "rect_params.json").read_text())
    info = {}
    for n, m, fa, ba, box in zip(cache["names"].tolist(), cache["methods"], cache["full_acc"], cache["best_acc"], cache["boxes"]):
        c = int(choose(np.array([fa]), np.array([ba]), *params[m])[0])
        info[n] = (fa < params[m][0], tuple(box) if c == 1 else None, None)  # (type B?, decoded rectangle, refined)
    f = OUT / f"refine_{split}.npz"
    if f.exists() and len(GRID["refine"]) > 1:                                                        # rect_refine.py: (best accuracy, box)
        z = np.load(f)
        for n, acc, box in zip(z["names"].tolist(), z["best_acc"], z["boxes"]):
            if n in info:
                info[n] = (info[n][0], info[n][1], (float(acc), tuple(int(v) for v in box)))
    return info


def score_set(names, probs, gts, info, prm, use_rect=True):
    return np.array([miou_image(post(probs[n], info[n][0], info[n][1] if use_rect else None, prm, info[n][2] if use_rect else None), gts[n]) for n in names])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="pixelseal=synth400,trustmark=synth400,stegastamp=real_noflip")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-refine", action="store_true", help="ignore rect_refine boxes (e.g. not computed on test yet)")
    a = ap.parse_args()
    if a.no_refine:
        GRID["refine"] = [2.0]
    sources = dict(kv.split("=") for kv in a.source.split(","))
    ds = LocalisationDataset("release", "train")
    gts = {m["image_name"]: ds.m128[m["image_name"]] for m in ds.meta}
    meta = {m["image_name"]: m for m in ds.meta}
    info = context("train")
    base_prm = {"thr": 0.5, "lcc": False, "snap": 1.01, "floor": 0.0}
    chosen, rows = {}, []
    for method in METHODS:
        probs = oof_probs(method, sources[method])
        names = sorted(n for n in probs if meta[n]["method"] == method)
        if not names:
            print(f"{method}: no OOF files yet"); continue
        full = np.array([bool(meta[n]["full_watermark"]) for n in names])
        fold = folds_of(np.array([method] * len(names)), full)
        held_scores, base_scores = np.zeros(len(names)), score_set(names, probs, gts, info, base_prm, use_rect=False)
        for k in range(5):
            tr, te = fold != k, fold == k
            if not te.any():
                continue
            best = max((dict(zip(GRID, v)) for v in itertools.product(*GRID.values())),
                       key=lambda prm: score_set([n for n, t in zip(names, tr) if t], probs, gts, info, prm).mean())
            held_scores[te] = score_set([n for n, t in zip(names, te) if t], probs, gts, info, best)
        chosen[method] = max((dict(zip(GRID, v)) for v in itertools.product(*GRID.values())),
                             key=lambda prm: score_set(names, probs, gts, info, prm).mean())
        d = held_scores - base_scores
        rows.append((method, len(names), base_scores.mean(), held_scores.mean(), d.mean(), d.std() / np.sqrt(len(d))))
        print(f"{method:10s} n={len(names)} raw {base_scores.mean():.4f} -> post-processed (nested CV) {held_scores.mean():.4f} "
              f"(paired gain {d.mean():+.4f} +- {d.std() / np.sqrt(len(d)):.4f}) | chosen {chosen[method]}", flush=True)
    if rows:
        print(f"overall raw {np.mean([r[2] for r in rows]):.4f} -> post {np.mean([r[3] for r in rows]):.4f}")
        (OUT / "post_params.json").write_text(json.dumps(chosen))
    if a.apply:
        import torch
        tinfo = context("test")
        test = LocalisationDataset("release", "test")
        tprobs = {}
        for m in METHODS:
            z = np.load(OUT / f"seg_probs_{m}_test.npz")
            tprobs.update(dict(zip(z["names"].tolist(), z["probs"].astype(np.float32))))
        names = [m["image_name"] for m in test.meta]
        masks = np.stack([post(tprobs[n], tinfo[n][0], tinfo[n][1], chosen[test.meta[i]["method"]], tinfo[n][2]) for i, n in enumerate(names)])
        assert masks.shape == (1020, 128, 128) and len(set(names)) == 1020
        torch.save({"image_names": names, "masks": torch.from_numpy(masks).bool()}, OUT / "submission_post.pt")
        print("wrote", OUT / "submission_post.pt", "| mean predicted watermarked share:", round(float(masks.mean()), 3))


if __name__ == "__main__":
    main()
