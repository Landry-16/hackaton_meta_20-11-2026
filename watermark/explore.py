"""Exploration on the train split, to pick a localisation method per watermark.
1. smoke: the three decoders load on GPU (and which ONNX provider StegaStamp really uses)
2. census: mask shapes per method and full_watermark (all 0, all 1, rectangle or not, components)
3. signal: bit accuracy vs the known message on the full image, the mask bounding box, windows inside / outside
   the mask; per-position logit agreement vs the mask for PixelSeal and TrustMark.
Run from the task data folder:  sbatch -J wm-explore jureca/job.sh watermark watermark/explore.py"""
import collections

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from wmloc.data import LocalisationDataset
from wmloc.utils import to_tensor
from wmloc.watermarks import load_watermark

N_SIGNAL = 24          # images per (method, full_watermark) group for the signal test
DEVICE = "cuda"


def bit_acc(wm, img: torch.Tensor, msg: np.ndarray) -> float:
    return float((wm.decode(img).cpu().numpy()[0] == msg).mean())


def crop(img: torch.Tensor, y0, y1, x0, x1) -> torch.Tensor:
    return img[:, :, y0:y1, x0:x1]


def census(ds):
    groups = collections.defaultdict(list)
    for it in ds:
        m = it["mask"]
        frac = m.mean()
        ys, xs = np.nonzero(m)
        box_fill = m.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1)) if len(ys) else float("nan")
        n_comp = ndimage.label(m)[1]
        n_holes = ndimage.label(1 - m)[1]
        groups[(it["method"], bool(it["full_watermark"]))].append((frac, box_fill, n_comp, n_holes))
    print("\n== census: method, full_watermark -> n, all0, all1, wm frac, wm area / bbox, components, clean components")
    for k, rows in sorted(groups.items()):
        a = np.array(rows, dtype=float)
        print(k, len(a), "all0 %d all1 %d" % ((a[:, 0] == 0).sum(), (a[:, 0] == 1).sum()),
              "frac %.2f" % a[:, 0].mean(),
              "fill med %.2f p10 %.2f" % (np.nanmedian(a[:, 1]), np.nanpercentile(a[:, 1], 10)),
              "comp med %d max %d" % (np.median(a[:, 2]), a[:, 2].max()),
              "clean comp med %d max %d" % (np.median(a[:, 3]), a[:, 3].max()))


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = scores.argsort().argsort().astype(float)
    return (ranks[labels == 1].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))


def signal(ds, wms):
    groups = collections.defaultdict(list)
    for it in ds:
        groups[(it["method"], bool(it["full_watermark"]))].append(it)
    print("\n== signal: method, full_watermark -> mean bit accuracy (0.5 = chance)")
    for (meth, full), items in sorted(groups.items()):
        wm = wms[meth]
        rows = collections.defaultdict(list)
        for it in items[:N_SIGNAL]:
            img, msg, m = to_tensor(it["image"]), it["message"], it["mask"]
            rows["full"].append(bit_acc(wm, img, msg))
            ys, xs = np.nonzero(m)
            if len(ys) and m.mean() < 1:
                rows["bbox"].append(bit_acc(wm, crop(img, ys.min(), ys.max() + 1, xs.min(), xs.max() + 1), msg))
            for w in (256, 128, 64):
                for y in range(0, 512, w):
                    for x in range(0, 512, w):
                        f = m[y:y + w, x:x + w].mean()
                        if f > 0.95 or f < 0.05:
                            rows[f"win{w}_{'in' if f > 0.95 else 'out'}"].append(
                                bit_acc(wm, crop(img, y, y + w, x, x + w), msg))
            if meth in ("pixelseal", "trustmark"):
                ppl = wm.features(img)["per_position_logits"][0]            # nbits x h x w
                agree = ((ppl > 0).float() == torch.tensor(msg, device=ppl.device)[:, None, None].float()).float().mean(0)
                up = F.interpolate(agree[None, None], size=(128, 128), mode="bilinear", align_corners=False)[0, 0]
                rows["ppl_auc"].append(auc(up.cpu().numpy().ravel(), it["mask128"].ravel()))
                rows["ppl_shape"] = [tuple(ppl.shape)]
        print((meth, full), {k: (v[0] if k == "ppl_shape" else "%.3f (n=%d)" % (np.nanmean(v), len(v)))
                             for k, v in sorted(rows.items())})


def main():
    ds = LocalisationDataset("release", "train")
    print("train images:", len(ds))
    wms = {name: load_watermark(name, device=DEVICE) for name in ("pixelseal", "trustmark", "stegastamp")}
    print("stegastamp ONNX providers:", wms["stegastamp"].sess.get_providers())
    print("cuda:", torch.cuda.get_device_name(0))
    census(ds)
    signal(ds, wms)


if __name__ == "__main__":
    main()
