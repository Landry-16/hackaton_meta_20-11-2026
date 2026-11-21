"""Synthetic training data for the watermark segmenter, built with the kit's own embedders.

Carriers: an image watermarked by one method is clean for the other two, so for method M we use the images of the
other two methods (train and test pixels, no labels involved). Each sample copies one of the real recipes:
  full     (type A) embed on the whole image, then restore 1 to 3 clean blobs (object removal / addition)
  rect     (type B) embed only inside a rectangle (outpainting: the original sits inside a larger image)
  object   (type B) embed a whole other carrier, paste one blob of it into this carrier (object transplantation)
Masks are 1 = watermarked, at 512 and on the official 128 grid.

  quick calibration (CPU):  python synth.py --method trustmark --n 12 --device cpu --calibrate
  full set on JURECA:       jureca/run.sh watermark watermark/synth.py --method trustmark --n 3000
Writes output/synth_<method>.npz (images uint8 N x 512 x 512 x 3, masks128 N x 128 x 128, kinds)."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from wmloc.data import LocalisationDataset
from wmloc.utils import downsample_mask, to_tensor

SIZE = 512
MAX_ASPECT = {"pixelseal": 16, "stegastamp": 16, "trustmark": 2}
NBITS = {"pixelseal": 256, "trustmark": 100, "stegastamp": 100}
OUT = Path("output")


def carriers(method):
    """Images watermarked by the two other methods: clean for this method."""
    out = []
    for split in ("train", "test"):
        ds = LocalisationDataset("release", split)
        out += [(ds, i) for i, m in enumerate(ds.meta) if m["method"] != method]
    return out


def blob(rng, area_frac):
    """A smooth random shape covering about area_frac of the image."""
    noise = ndimage.gaussian_filter(rng.standard_normal((SIZE, SIZE)), sigma=rng.uniform(12, 40))
    cy, cx = rng.uniform(0.15, 0.85, 2) * SIZE
    yy, xx = np.mgrid[:SIZE, :SIZE]
    r = np.hypot((yy - cy) / rng.uniform(0.6, 1.6), (xx - cx) / rng.uniform(0.6, 1.6))
    field = -r / SIZE + 0.35 * noise / noise.std()
    return (field >= np.quantile(field, 1 - area_frac)).astype(np.uint8)


def rect(rng, method):
    area = np.exp(rng.uniform(np.log(0.01), np.log(0.35))) * SIZE * SIZE
    aspect = np.exp(rng.uniform(0, np.log(min(MAX_ASPECT[method], 6))))
    h = int(np.clip(np.sqrt(area * aspect), 24, SIZE))
    w = int(np.clip(area / h, 24, SIZE))
    if rng.random() < 0.5:
        h, w = w, h
    y0, x0 = rng.integers(0, SIZE - h + 1), rng.integers(0, SIZE - w + 1)
    return y0, y0 + h, x0, x0 + w


def make_sample(wm, method, img, other, rng):
    bits = rng.integers(0, 2, NBITS[method])
    kind = rng.choice(["full", "rect", "object"], p=[0.5, 0.3, 0.2])
    if kind == "full":
        out, mask = np.array(wm.embed(img, bits)), np.ones((SIZE, SIZE), np.uint8)
        for _ in range(rng.integers(1, 4)):
            hole = blob(rng, rng.uniform(0.02, 0.15))
            out[hole == 1], mask[hole == 1] = img[hole == 1], 0
    elif kind == "rect":
        y0, y1, x0, x1 = rect(rng, method)
        out, mask = img.copy(), np.zeros((SIZE, SIZE), np.uint8)
        # outpainting: a whole watermarked image is shrunk and placed inside a bigger one
        small = Image.fromarray(np.array(wm.embed(other, bits))).resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
        out[y0:y1, x0:x1], mask[y0:y1, x0:x1] = np.asarray(small), 1
    else:
        donor = np.array(wm.embed(other, bits))
        mask = blob(rng, rng.uniform(0.02, 0.25))
        out = img.copy()
        out[mask == 1] = donor[mask == 1]
    return out, mask, kind, bits


RESAMPLERS = (Image.Resampling.LANCZOS, Image.Resampling.BICUBIC, Image.Resampling.BILINEAR)
_HOLES = None


def real_holes():
    """Hole shapes of the real type A train masks (0 = clean), reused with random flips and shifts."""
    global _HOLES
    if _HOLES is None:
        ds = LocalisationDataset("release", "train")
        _HOLES = [1 - ds[i]["mask"] for i, m in enumerate(ds.meta) if m["full_watermark"]]
    return _HOLES


def embed_degraded(wm, img, bits, rng):
    """Embed like the real pipeline seems to: on a slightly larger canvas then cropped back, then a light blur."""
    m = int(rng.integers(0, 33))
    if m:
        canvas = np.pad(img, ((m, m), (m, m), (0, 0)), mode="reflect")
        out = np.array(wm.embed(canvas, bits))[m:m + img.shape[0], m:m + img.shape[1]]
    else:
        out = np.array(wm.embed(img, bits))
    sigma = rng.uniform(0, 0.8)
    if sigma > 0.05:
        out = np.clip(ndimage.gaussian_filter(out.astype(np.float32), sigma=(sigma, sigma, 0)), 0, 255).astype(np.uint8)
    return out


def make_sample_v2(wm, method, img, rng):
    """Same image only (no foreign content), real hole shapes, realistic degradation, aspect up to the task limit."""
    bits = rng.integers(0, 2, NBITS[method])
    kind = rng.choice(["full", "rect", "object"], p=[0.5, 0.3, 0.2])
    if kind == "full":
        out, mask = embed_degraded(wm, img, bits, rng), np.ones((SIZE, SIZE), np.uint8)
        hole = real_holes()[rng.integers(len(real_holes()))]
        hole = np.roll(hole[:, ::-1] if rng.random() < 0.5 else hole, rng.integers(-64, 65, 2), axis=(0, 1))
        out[hole == 1], mask[hole == 1] = img[hole == 1], 0
    elif kind == "rect":
        area = np.exp(rng.uniform(np.log(0.01), np.log(0.35))) * SIZE * SIZE
        aspect = np.exp(rng.uniform(0, np.log(MAX_ASPECT[method])))
        h = int(np.clip(np.sqrt(area * aspect), 16, SIZE))
        w = int(np.clip(area / h, 16, SIZE))
        if rng.random() < 0.5:
            h, w = w, h
        y0, x0 = rng.integers(0, SIZE - h + 1), rng.integers(0, SIZE - w + 1)
        # outpainting in place: this part of the image, watermarked at full size then shrunk back where it was
        big = np.asarray(Image.fromarray(np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w])).resize((SIZE, SIZE), Image.Resampling.BICUBIC))
        small = Image.fromarray(embed_degraded(wm, big, bits, rng)).resize((w, h), RESAMPLERS[rng.integers(3)])
        out, mask = img.copy(), np.zeros((SIZE, SIZE), np.uint8)
        out[y0:y0 + h, x0:x0 + w], mask[y0:y0 + h, x0:x0 + w] = np.asarray(small), 1
    else:
        wmk = embed_degraded(wm, img, bits, rng)
        mask = blob(rng, rng.uniform(0.02, 0.25))
        out = img.copy()
        out[mask == 1] = wmk[mask == 1]
    return out, mask, kind, bits


def calibrate(wm, method, samples):
    """Do synthetic samples decode like the real ones? full images about 0.96, exact rectangles like the real share."""
    rows = {"full": [], "rect": [], "object": []}
    for img, mask, kind, bits in samples:
        if kind == "rect":
            ys, xs = np.nonzero(mask)
            img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        acc = float((wm.decode(to_tensor(np.ascontiguousarray(img))).cpu().numpy()[0] == bits).mean())
        rows[kind].append(acc)
    for k, v in rows.items():
        if v:
            print(f"calibration {method} {k}: n={len(v)} mean bit acc {np.mean(v):.3f}, share >= 0.85: {np.mean(np.array(v) >= 0.85):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(NBITS))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--v2", action="store_true", help="recipe v2: same image only, real hole shapes, degradation")
    a = ap.parse_args()
    from wmloc.watermarks import load_watermark
    print(f"device {a.device}, cuda available {torch.cuda.is_available()}", flush=True)
    wm = load_watermark(a.method, device=a.device)
    pool, rng = carriers(a.method), np.random.default_rng(a.seed)
    images, masks, kinds, kept = [], [], [], []
    t0 = time.time()
    for i in range(a.n):
        (ds1, i1), (ds2, i2) = (pool[j] for j in rng.choice(len(pool), 2, replace=False))
        if a.v2:
            img, mask, kind, bits = make_sample_v2(wm, a.method, ds1[i1]["image"], rng)
        else:
            img, mask, kind, bits = make_sample(wm, a.method, ds1[i1]["image"], ds2[i2]["image"], rng)
        images.append(img)
        masks.append(downsample_mask(mask))
        kinds.append(kind)
        if a.calibrate:
            kept.append((img, mask, kind, bits))
        if (i + 1) % 100 == 0 or i + 1 == a.n:
            print(f"{a.method} {i + 1}/{a.n} samples, {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / f"synth{'v2' if a.v2 else ''}_{a.method}{'_quick' if a.calibrate else ''}.npz",
             images=np.stack(images), masks128=np.stack(masks), kinds=np.array(kinds))
    if a.calibrate:
        calibrate(wm, a.method, kept)


if __name__ == "__main__":
    main()
