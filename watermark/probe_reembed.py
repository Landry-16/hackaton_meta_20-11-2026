"""Probe: re-embed the known message and correlate locally.
We know the message and have the exact embedder. Re-embedding the same message gives an estimate of the watermark
pattern r = embed(img, bits) - img. Where the image is already watermarked it contains that pattern, so the local
correlation between the image's high frequencies and r should be high; on clean parts it should be near zero.
AUC of the local correlation against the true 128 grid mask: 0.5 = useless, near 1 = the idea works.
  python probe_reembed.py --n 10"""
import argparse

import numpy as np
from scipy import ndimage

from rect_search import auc_of
from wmloc.data import LocalisationDataset
from wmloc.utils import downsample_mask
from wmloc.watermarks import load_watermark


def highpass(x, sigma=1.5):
    return x - ndimage.gaussian_filter(x, sigma=(sigma, sigma, 0))


def local_corr(a, b, win=16):
    """Normalized correlation of a and b in a win x win window around every pixel (summed over channels)."""
    f = lambda z: ndimage.uniform_filter(z, size=win)
    ab, aa, bb = f((a * b).sum(-1)), f((a * a).sum(-1)), f((b * b).sum(-1))
    return ab / np.sqrt(aa * bb + 1e-8)


def corr_map(wm, image, bits, win=16):
    img = image.astype(np.float32)
    r = np.asarray(wm.embed(image, bits)).astype(np.float32) - img
    c = local_corr(highpass(img), highpass(r), win)
    return c.reshape(128, 4, 128, 4).mean((1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="images per method")
    ap.add_argument("--win", type=int, default=16)
    a = ap.parse_args()
    for method in ("trustmark", "stegastamp", "pixelseal"):
        wm = load_watermark(method, device="cpu")
        ds = LocalisationDataset("release", "train", methods=[method])
        rows = []
        for i in range(a.n):
            it = ds[i]
            m = corr_map(wm, it["image"], it["message"], a.win)
            rows.append((auc_of(m.ravel(), it["mask128"].ravel()), bool(it["full_watermark"]),
                         m[it["mask128"] == 1].mean() if it["mask128"].any() else np.nan, m[it["mask128"] == 0].mean()))
        r = np.array(rows, dtype=float)
        print(f"{method}: AUC {np.nanmean(r[:, 0]):.3f} (type A {np.nanmean(r[r[:, 1] == 1, 0]):.3f}, type B {np.nanmean(r[r[:, 1] == 0, 0]):.3f}) "
              f"| mean corr watermarked {np.nanmean(r[:, 2]):.3f} vs clean {np.nanmean(r[:, 3]):.3f}", flush=True)


if __name__ == "__main__":
    main()
