"""Synthetic recipe v4: v3 (residual only, native-size rectangles) + the real edit recipes with foreign content, on
outside images. Carriers are COCO val2017 photos (read straight from the zip) mixed with the other methods' images;
hole and object shapes are real COCO object masks instead of random blobs.
  full    (50 %) whole image watermarked, then 1 to 3 object-shaped holes: the clean original restored (60 %,
          object removal) or another image's pixels pasted (40 %, object addition)
  rect    (25 %) a rectangle watermarked at its own scale: this image's crop (50 %) or another image's crop (50 %)
  object  (25 %) an object cut from a fully watermarked donor (another image 60 %, this one 40 %) kept at the SAME
          coordinates (object transplantation: moving it would break the watermark phase)
Saves images, masks, kinds, the measured full-image decode accuracy and the message bits.
  quick check:  python synth4.py --method trustmark --n 12 --device cpu --suffix _quick
  queue:        jureca/q.sh gen4 watermark/synth4.py --method trustmark --n 6000
Needs coco/val2017.zip and coco/instances_val2017.json in the task data folder (2 files, nothing unpacked)."""
import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from synth import MAX_ASPECT, NBITS, SIZE, carriers, residual
from wmloc.utils import downsample_mask, to_tensor

OUT = Path("output")


class Coco:
    def __init__(self, root="coco"):
        root = Path(root)
        self.zip = zipfile.ZipFile(root / "val2017.zip")
        self.files = [n for n in self.zip.namelist() if n.endswith(".jpg")]
        ann = json.loads((root / "instances_val2017.json").read_text())
        self.size = {im["id"]: (im["height"], im["width"]) for im in ann["images"]}
        self.anns = [a for a in ann["annotations"] if not a["iscrowd"] and a["area"] > 400]

    def image(self, rng):
        """A random photo, random square crop, 512 x 512."""
        im = Image.open(io.BytesIO(self.zip.read(self.files[rng.integers(len(self.files))]))).convert("RGB")
        w, h = im.size
        s = min(w, h)
        x0, y0 = rng.integers(0, w - s + 1), rng.integers(0, h - s + 1)
        return np.asarray(im.crop((x0, y0, x0 + s, y0 + s)).resize((SIZE, SIZE), Image.Resampling.BICUBIC), dtype=np.uint8)

    def shape(self, rng):
        """A random object mask cropped to its bounding box (bool array)."""
        from pycocotools import mask as cm
        while True:
            a = self.anns[rng.integers(len(self.anns))]
            h, w = self.size[a["image_id"]]
            rle = cm.frPyObjects(a["segmentation"], h, w)
            m = cm.decode(cm.merge(rle) if isinstance(rle, list) else rle).astype(bool)
            ys, xs = np.nonzero(m)
            if len(ys) > 100:
                return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def place(shape, area_frac, rng):
    """The object mask scaled to about area_frac of the image, at a random position (512 x 512 uint8)."""
    scale = np.sqrt(area_frac * SIZE * SIZE / max(shape.sum(), 1))
    h = int(np.clip(shape.shape[0] * scale, 8, SIZE))
    w = int(np.clip(shape.shape[1] * scale, 8, SIZE))
    small = np.asarray(Image.fromarray(shape.astype(np.uint8) * 255).resize((w, h), Image.Resampling.BILINEAR)) > 127
    y0, x0 = rng.integers(0, SIZE - h + 1), rng.integers(0, SIZE - w + 1)
    out = np.zeros((SIZE, SIZE), np.uint8)
    out[y0:y0 + h, x0:x0 + w] = small
    return out


def log_area(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def make_sample_v4(wm, method, img, other, coco, rng):
    bits = rng.integers(0, 2, NBITS[method])
    base = img.astype(np.float32)
    u = rng.random()
    if u < 0.5:                                                    # full image, object-shaped holes
        res = residual(wm, img, bits, rng)
        hole = np.zeros((SIZE, SIZE), np.uint8)
        for _ in range(1 if rng.random() < 0.7 else int(rng.integers(2, 4))):
            hole |= place(coco.shape(rng), log_area(rng, 0.004, 0.3), rng)
        if hole.mean() > 0.49 or hole.sum() < 400:
            hole = place(coco.shape(rng), 0.05, rng)
        out = base + res
        fill = base if rng.random() < 0.6 else other.astype(np.float32)
        out[hole == 1] = fill[hole == 1]
        return _u8(out), 1 - hole, "full", bits
    if u < 0.75:                                                   # rectangle watermarked at its own scale
        area = log_area(rng, 0.004, 0.45) * SIZE * SIZE
        aspect = np.exp(rng.uniform(0, np.log(MAX_ASPECT[method] if rng.random() < 0.8 else min(MAX_ASPECT[method], 1.5))))
        h = int(np.clip(np.sqrt(area * aspect), 24, SIZE - 8))
        w = int(np.clip(area / h, 24, SIZE - 8))
        if rng.random() < 0.5:
            h, w = w, h
        y0, x0 = rng.integers(0, SIZE - h + 1), rng.integers(0, SIZE - w + 1)
        src = img if rng.random() < 0.5 else other
        d = int(rng.integers(0, 5))                               # the embedding frame is 0 to 4 px larger than the mask
        Y0, Y1, X0, X1 = max(0, y0 - d), min(SIZE, y0 + h + d), max(0, x0 - d), min(SIZE, x0 + w + d)
        crop = np.ascontiguousarray(src[Y0:Y1, X0:X1])
        wmk = crop.astype(np.float32) + residual(wm, crop, bits, rng, margins=(0, 0, 0, 0))
        out, mask = base.copy(), np.zeros((SIZE, SIZE), np.uint8)
        out[y0:y0 + h, x0:x0 + w] = wmk[y0 - Y0:y0 - Y0 + h, x0 - X0:x0 - X0 + w]
        mask[y0:y0 + h, x0:x0 + w] = 1
        return _u8(out), mask, "rect", bits
    donor = other if rng.random() < 0.6 else img                   # object from a fully watermarked donor, kept in place
    wmk = donor.astype(np.float32) + residual(wm, donor, bits, rng)
    mask = place(coco.shape(rng), log_area(rng, 0.004, 0.3), rng)
    out = base.copy()
    out[mask == 1] = wmk[mask == 1]
    return _u8(out), mask, "object", bits


def _u8(x):
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(NBITS))
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    from wmloc.watermarks import load_watermark
    print(f"device {a.device}, cuda available {torch.cuda.is_available()}", flush=True)
    wm = load_watermark(a.method, device=a.device)
    coco, pool, rng = Coco(), carriers(a.method), np.random.default_rng(a.seed)
    pick = lambda: coco.image(rng) if rng.random() < 0.7 else (lambda p: p[0][p[1]]["image"])(pool[rng.integers(len(pool))])
    images, masks, kinds, accs, allbits, t0 = [], [], [], [], [], time.time()
    for i in range(a.n):
        img, mask, kind, bits = make_sample_v4(wm, a.method, pick(), pick(), coco, rng)
        images.append(img); masks.append(downsample_mask(mask)); kinds.append(kind); allbits.append(bits)
        accs.append(float((wm.decode(to_tensor(img).to(a.device)).cpu().numpy()[0] == bits).mean()))
        if (i + 1) % 100 == 0 or i + 1 == a.n:
            print(f"{a.method} {i + 1}/{a.n} samples, {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(exist_ok=True)
    tmp = OUT / f"synthv4{a.suffix}_{a.method}.tmp.npz"
    np.savez(tmp, images=np.stack(images), masks128=np.stack(masks), kinds=np.array(kinds),
             acc=np.array(accs, np.float32), bits=np.array(allbits, np.uint8))
    tmp.rename(OUT / f"synthv4{a.suffix}_{a.method}.npz")
    k, acc, m = np.array(kinds), np.array(accs), np.stack(masks)
    print("measured full-image decode:", {t: round(float(acc[k == t].mean()), 3) for t in ("full", "rect", "object")},
          "| watermarked share:", {t: round(float(m[k == t].mean()), 3) for t in ("full", "rect", "object")}, flush=True)


if __name__ == "__main__":
    main()
