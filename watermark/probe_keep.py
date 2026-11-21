"""Two probes for local watermark evidence, scored by AUC against the true 128 grid (0.5 = useless).
keep:    keep one W x W window sharp, blur the rest, decode the whole picture (right scale and geometry);
         each cell gets the mean decoding margin of the windows that cover it.
remove:  TrustMark only: |image - TrustMark's own watermark remover output|, high-passed, as local energy.
  python probe_keep.py --n 8 --win 128 --stride 64"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from rect_search import auc_of
from wmloc.data import LocalisationDataset
from wmloc.utils import to_tensor
from wmloc.watermarks import load_watermark


def keep_map(wm, it, win, stride, device):
    img = to_tensor(it["image"]).to(device)
    blur = F.avg_pool2d(img, 31, stride=1, padding=15, count_include_pad=False)
    sign = torch.tensor(it["message"].astype(np.float32) * 2 - 1, device=device)
    starts = list(range(0, 512 - win + 1, stride))
    boxes = [(y, x) for y in starts for x in starts]
    scores = []
    for i in range(0, len(boxes), 16):
        batch = []
        for y, x in boxes[i:i + 16]:
            v = blur.clone()
            v[:, :, y:y + win, x:x + win] = img[:, :, y:y + win, x:x + win]
            batch.append(v)
        scores.append(torch.tanh(wm.logits(torch.cat(batch)).to(device) * sign).mean(1).cpu())
    scores = torch.cat(scores).numpy()
    acc, cnt = np.zeros((512, 512)), np.zeros((512, 512))
    for (y, x), s in zip(boxes, scores):
        acc[y:y + win, x:x + win] += s
        cnt[y:y + win, x:x + win] += 1
    return (acc / np.maximum(cnt, 1)).reshape(128, 4, 128, 4).mean((1, 3))


def remove_map(wm, it):
    removed = np.asarray(wm.tm.remove_watermark(Image.fromarray(it["image"])).convert("RGB")).astype(np.float32)
    d = np.abs(it["image"].astype(np.float32) - removed).sum(-1)
    return ndimage.uniform_filter(d, 9).reshape(128, 4, 128, 4).mean((1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--win", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    for method in ("trustmark", "stegastamp", "pixelseal"):
        wm = load_watermark(method, device=a.device)
        ds = LocalisationDataset("release", "train", methods=[method])
        res = {"keep": [], "remove": []}
        for i in range(a.n):
            it = ds[i]
            lab = it["mask128"].ravel()
            res["keep"].append((auc_of(keep_map(wm, it, a.win, a.stride, a.device).ravel(), lab), it["full_watermark"]))
            if method == "trustmark":
                res["remove"].append((auc_of(remove_map(wm, it).ravel(), lab), it["full_watermark"]))
        for k, v in res.items():
            if v:
                r = np.array(v, dtype=float)
                print(f"{method} {k}: AUC {np.nanmean(r[:, 0]):.3f} (type A {np.nanmean(r[r[:, 1] == 1, 0]):.3f}, "
                      f"type B {np.nanmean(r[r[:, 1] == 0, 0]):.3f}) over {len(r)} images", flush=True)


if __name__ == "__main__":
    main()
