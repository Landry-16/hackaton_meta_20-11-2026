"""Type B: refine decodable rectangles seeded by the segmenter instead of a fixed grid.
For each type B image, the bounding box of every predicted component (probability >= 0.5, plus the largest one at a
lower threshold) is a seed; each edge then moves by 16, 8, 4, 2 px while the crop's bit accuracy on the known message
improves. The best box and its accuracy are saved; postproc.py uses it when the accuracy passes a CV-chosen threshold.
  python rect_refine.py --split train --probs oof         (seeds from the out-of-fold predictions)
  python rect_refine.py --split test --probs test         (seeds from seg_probs_<method>_test.npz)
Writes output/refine_<split>.npz (names, best_acc, boxes)."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from postproc import oof_probs
from rect_search import METHODS, MAX_ASPECT, crop_scores, load_cache
from wmloc.data import LocalisationDataset
from wmloc.utils import to_tensor

OUT = Path("output")
SIZE = 512


def ok(box, method):
    y0, y1, x0, x1 = box
    h, w = y1 - y0, x1 - x0
    return 0 <= y0 and y1 <= SIZE and 0 <= x0 and x1 <= SIZE and min(h, w) >= 8 and max(h, w) / min(h, w) <= MAX_ASPECT[method]


def seeds(prob):
    out = []
    for thr in (0.5, 0.3):
        lab, n = ndimage.label(prob >= thr)
        for sl in ndimage.find_objects(lab):
            if sl is not None:
                ys, xs = sl
                out.append((ys.start * 4, ys.stop * 4, xs.start * 4, xs.stop * 4))   # 128 grid -> 512 px
    return list(dict.fromkeys(out))[:6]


def refine(wm, method, img, sign, box):
    best = float(crop_scores(wm, method, img, [box], sign)[0])
    for step in (16, 8, 4, 2):
        improved = True
        while improved:
            moves = [tuple(v + d if k == e else v for k, v in enumerate(box)) for e in range(4) for d in (-step, step)]
            moves = [m for m in moves if ok(m, method)]
            if not moves:
                break
            s = crop_scores(wm, method, img, moves, sign)
            improved = float(s.max()) > best
            if improved:
                box, best = moves[int(s.argmax())], float(s.max())
    return box, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--probs", default="oof", choices=["oof", "test"])
    ap.add_argument("--source", default="pixelseal=synth400,trustmark=synth400,stegastamp=real")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from wmloc.watermarks import load_watermark
    sources = dict(kv.split("=") for kv in a.source.split(","))
    probs = {}
    for m in METHODS:
        if a.probs == "oof":
            probs.update(oof_probs(m, sources[m]))
        else:
            z = np.load(OUT / f"seg_probs_{m}_test.npz")
            probs.update(dict(zip(z["names"].tolist(), z["probs"].astype(np.float32))))
    cache = load_cache(a.split)
    full_acc = dict(zip(cache["names"].tolist(), cache["full_acc"]))
    wms = {m: load_watermark(m, device=a.device) for m in METHODS}
    ds = LocalisationDataset("release", a.split)
    rows, t0 = {"names": [], "best_acc": [], "boxes": []}, time.time()
    n = min(a.limit or len(ds), len(ds))
    for i in range(n):
        meta = ds.meta[i]
        name, m = meta["image_name"], meta["method"]
        if name not in probs or full_acc[name] >= 0.75:          # type A: nothing to refine
            continue
        it = ds[i]
        img = to_tensor(it["image"]).to(a.device)
        sign = torch.tensor(it["message"].astype(np.float32) * 2 - 1, device=a.device)
        best_box, best = (0, SIZE, 0, SIZE), 0.0
        for s in seeds(probs[name]):
            if ok(s, m):
                box, acc = refine(wms[m], m, img, sign, s)
                if acc > best:
                    best_box, best = box, acc
        rows["names"].append(name); rows["best_acc"].append(best); rows["boxes"].append(best_box)
        if len(rows["names"]) % 25 == 0:
            print(f"{a.split}: {len(rows['names'])} type B images refined, {time.time() - t0:.0f} s", flush=True)
    np.savez(OUT / f"refine_{a.split}.npz", **{k: np.array(v) for k, v in rows.items()})
    acc = np.array(rows["best_acc"])
    print(f"wrote refine_{a.split}.npz: {len(acc)} images, share with a crop decoding >= 0.85: {(acc >= 0.85).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
