"""Baseline v1: tell the two regimes apart from the full-image decode, then predict a constant mask per regime.
Exploration showed: full_watermark images (about 90 % watermarked, with holes) decode at 0.95+ bit accuracy on the
whole image, the others (a small rectangle, about 8 %) decode near chance. So:
  full-image bit accuracy >= threshold -> all ones, else -> all zeros.
The threshold is picked on train with the official score. Writes submission_regime.pt for the test split.
Run:  sbatch -J wm-regime jureca/job.sh watermark watermark/regime.py"""
import numpy as np
import torch

from wmloc.data import LocalisationDataset
from wmloc.utils import score, to_tensor
from wmloc.watermarks import load_watermark

METHODS = ("pixelseal", "trustmark", "stegastamp")


def full_accuracy(ds, wms):
    accs = []
    for it in ds:
        bits = wms[it["method"]].decode(to_tensor(it["image"])).cpu().numpy()[0]
        accs.append(float((bits == it["message"]).mean()))
    return np.array(accs)


def masks_for(accs, thr):
    return [np.full((128, 128), int(a >= thr), np.uint8) for a in accs]


def main():
    wms = {m: load_watermark(m, device="cuda") for m in METHODS}
    train, test = LocalisationDataset("release", "train"), LocalisationDataset("release", "test")
    tr_acc = full_accuracy(train, wms)
    gts = [it["mask128"] for it in train]
    full = np.array([bool(it["full_watermark"]) for it in train])
    methods = np.array([it["method"] for it in train])

    thr = {}
    for m in METHODS:
        sel = methods == m
        best = max(np.arange(0.55, 0.96, 0.01), key=lambda t: score(masks_for(tr_acc[sel], t), [g for g, s in zip(gts, sel) if s]))
        thr[m] = float(best)
        regime_acc = ((tr_acc[sel] >= best) == full[sel]).mean()
        print(f"{m}: threshold {best:.2f}, regime accuracy {regime_acc:.3f}, "
              f"train score {score(masks_for(tr_acc[sel], best), [g for g, s in zip(gts, sel) if s]):.4f}")
    preds = [np.full((128, 128), int(a >= thr[m]), np.uint8) for a, m in zip(tr_acc, methods)]
    print(f"train score overall: {score(preds, gts):.4f}")

    te_acc = full_accuracy(test, wms)
    names = [it["image_name"] for it in test]
    masks = np.stack([np.full((128, 128), int(a >= thr[it["method"]]), np.uint8) for a, it in zip(te_acc, test)])
    assert masks.shape == (1020, 128, 128) and len(set(names)) == 1020
    torch.save({"image_names": names, "masks": torch.from_numpy(masks).bool()}, "submission_regime.pt")
    np.save("test_full_accuracy.npy", te_acc)
    print("test predicted full-watermark share:", masks.reshape(1020, -1)[:, 0].mean(), "-> wrote submission_regime.pt")


if __name__ == "__main__":
    main()
