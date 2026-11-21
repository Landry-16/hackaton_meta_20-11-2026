"""Exhaustive search of the exact embedding rectangle for type B images (PixelSeal, TrustMark).
A rectangle watermarked on its own decodes the known message when cropped EXACTLY. For each image predicted type B:
  1. enumerate ~20-28k boxes (sides 28..512, positions every max(8, side // 3) px plus edge-aligned ones, aspect
     limit of the method), crop and resize them all at once with roi_align to the decoder resolution, score each by
     its mean signed logit (logit x message sign);
  2. keep the 6 best distinct boxes (IoU < 0.5) and hill-climb each edge by 16, 8, 4, 2 px (exact crop + kit resize);
  3. keep the best box with its bit accuracy.
Output: output/refine_<split>.npz (names, best_acc, boxes), read by postproc.py, which chooses the acceptance
threshold by nested CV (no false detection allowed to pay off).
  python global_rect.py --split train --methods trustmark --shard 0/1
  queue:  jureca/q.sh grect watermark/global_rect.py --split test --methods pixelseal --shard 0/2"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou, roi_align

from rect_search import MAX_ASPECT, crop_scores, load_cache
from wmloc.data import LocalisationDataset
from wmloc.utils import to_tensor

OUT = Path("output")
SIZE = 512
SIDES = (28, 40, 56, 80, 112, 160, 224, 320, 448, 512)
RES = {"pixelseal": 256, "trustmark": 245}
REGIME_THR = 0.75          # full-image bit accuracy below this = type B (all methods: 0.73-0.95 thresholds, 0.53 typical B)


def candidates(method):
    boxes = []
    for h in SIDES:
        for w in SIDES:
            if max(h, w) / min(h, w) > MAX_ASPECT[method]:
                continue
            ys = sorted(set(list(range(0, SIZE - h + 1, max(8, h // 3))) + [SIZE - h]))
            xs = sorted(set(list(range(0, SIZE - w + 1, max(8, w // 3))) + [SIZE - w]))
            boxes += [(y, y + h, x, x + w) for y in ys for x in xs]
    return np.array(boxes)


@torch.no_grad()
def coarse_scores(wm, method, img, sign, boxes, batch=384):
    res = RES[method]
    out = []
    for i in range(0, len(boxes), batch):
        b = torch.tensor(boxes[i:i + batch], dtype=torch.float32, device=img.device)
        rois = torch.stack([torch.zeros(len(b), device=img.device), b[:, 2], b[:, 0], b[:, 3], b[:, 1]], 1)   # (idx, x1, y1, x2, y2)
        crops = roi_align(img, rois, output_size=(res, res), spatial_scale=1.0, sampling_ratio=0, aligned=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=img.is_cuda):
            out.append((wm.logits(crops.clamp(0, 1)).float().to(sign.device) * sign).mean(1).cpu())
    return torch.cat(out).numpy()


def valid(box, method):
    y0, y1, x0, x1 = box
    h, w = y1 - y0, x1 - x0
    return 0 <= y0 and y1 <= SIZE and 0 <= x0 and x1 <= SIZE and min(h, w) >= 16 and max(h, w) / min(h, w) <= MAX_ASPECT[method]


def hill_climb(wm, method, img, sign, box):
    best = float(crop_scores(wm, method, img, [tuple(box)], sign)[0])
    box = tuple(int(v) for v in box)
    for step in (16, 8, 4, 2):
        for _ in range(12):
            moves = [tuple(v + d if k == e else v for k, v in enumerate(box)) for e in range(4) for d in (-step, step)]
            moves = [m for m in moves if valid(m, method)]
            s = crop_scores(wm, method, img, moves, sign)
            if not moves or float(s.max()) <= best:
                break
            box, best = moves[int(s.argmax())], float(s.max())
    return box, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--methods", default="trustmark,pixelseal")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th image starting at i (parallel GPUs)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    from wmloc.watermarks import load_watermark
    k, n = (int(v) for v in a.shard.split("/"))
    methods = a.methods.split(",")
    cache = load_cache(a.split)
    full_acc = dict(zip(cache["names"].tolist(), cache["full_acc"]))
    ds = LocalisationDataset("release", a.split)
    todo = [i for i, m in enumerate(ds.meta) if m["method"] in methods and full_acc[m["image_name"]] < REGIME_THR][k::n]
    todo = todo[:a.limit] if a.limit else todo
    wms = {m: load_watermark(m, device=a.device) for m in methods}
    cands = {m: candidates(m) for m in methods}
    print(f"{len(todo)} type B images, candidates per image: " + ", ".join(f"{m} {len(c)}" for m, c in cands.items()), flush=True)
    rows, t0 = {"names": [], "best_acc": [], "boxes": [], "score": []}, time.time()
    for j, i in enumerate(todo):
        it = ds[i]
        m = it["method"]
        img = to_tensor(it["image"]).to(a.device)
        sign = torch.tensor(it["message"].astype(np.float32) * 2 - 1, device=a.device)
        sc = coarse_scores(wms[m], m, img, sign, cands[m])
        top = np.argsort(-sc)[:200]
        tb = torch.tensor(cands[m][top][:, [2, 0, 3, 1]], dtype=torch.float32)   # x1, y1, x2, y2 for box_iou
        keep = []
        for q in range(len(top)):
            if all(box_iou(tb[q:q + 1], tb[r:r + 1]).item() < 0.5 for r in keep):
                keep.append(q)
            if len(keep) == 6:
                break
        best_box, best = None, -1.0
        for q in keep:
            box, acc = hill_climb(wms[m], m, img, sign, cands[m][top[q]])
            if acc > best:
                best_box, best = box, acc
        rows["names"].append(it["image_name"]); rows["best_acc"].append(best); rows["boxes"].append(best_box); rows["score"].append(float(sc[top[0]]))
        if (j + 1) % 10 == 0 or j + 1 == len(todo):
            print(f"{j + 1}/{len(todo)} images, {time.time() - t0:.0f} s, found >= 0.8 so far: {np.mean(np.array(rows['best_acc']) >= 0.8):.2f}", flush=True)
    out = OUT / f"grect_{a.split}_{'-'.join(methods)}_{k}of{n}.npz"
    np.savez(out, **{key: np.array(v) for key, v in rows.items()})
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
