"""Probe: can the decoder itself point at the holes of type A images?
Blur one cell of an 8x8 grid at a time and measure how much the decoding margin on the known message drops.
Blurring a watermarked cell should remove signal (big drop); blurring a hole (clean cell) should change nothing.
AUC of the drop against the true cell label: about 0.5 = useless, close to 1 = the idea works.
  python probe_judge.py --n 12 --device cpu"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from rect_search import auc_of
from wmloc.data import LocalisationDataset
from wmloc.utils import to_tensor
from wmloc.watermarks import load_watermark

GRID, CELL = 8, 64


def margin(wm, images, sign):
    return (torch.tanh(wm.logits(images).to(sign.device) * sign)).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="type A images per method")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    for method in ("trustmark", "pixelseal", "stegastamp"):
        wm = load_watermark(method, device=a.device)
        ds = LocalisationDataset("release", "train", methods=[method])
        aucs, holes = [], 0
        for it in (ds[i] for i in range(len(ds))):
            if not it["full_watermark"]:
                continue
            img = to_tensor(it["image"]).to(a.device)
            blurred = F.avg_pool2d(img, 15, stride=1, padding=7, count_include_pad=False)
            sign = torch.tensor(it["message"].astype(np.float32) * 2 - 1, device=a.device)
            variants = []
            for gy in range(GRID):
                for gx in range(GRID):
                    v = img.clone()
                    v[:, :, gy * CELL:(gy + 1) * CELL, gx * CELL:(gx + 1) * CELL] = blurred[:, :, gy * CELL:(gy + 1) * CELL, gx * CELL:(gx + 1) * CELL]
                    variants.append(v)
            base = margin(wm, img, sign)
            drops = torch.cat([base - margin(wm, torch.cat(variants[i:i + 16]), sign) for i in range(0, 64, 16)]).cpu().numpy()
            label = it["mask"].reshape(GRID, CELL, GRID, CELL).mean((1, 3)).ravel() >= 0.5
            if 0 < label.sum() < 64:
                aucs.append(auc_of(drops, label.astype(int)))
                holes += int((~label).sum())
            if len(aucs) >= a.n:
                break
        print(f"{method}: AUC of the decoder drop vs watermarked cells = {np.nanmean(aucs):.3f} "
              f"(min {np.nanmin(aucs):.2f}, max {np.nanmax(aucs):.2f}) over {len(aucs)} images, {holes} hole cells", flush=True)


if __name__ == "__main__":
    main()
