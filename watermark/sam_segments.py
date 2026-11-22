"""Class-agnostic object segments of every image, from SAM (ViT-H, automatic mask generator), stored at 128 x 128
with the official downsampling rule. The watermark heads decide WHERE roughly; these segments give the exact shape.
  queue:  jureca/q.sh sam watermark/sam_segments.py --split train --shard 0 --of 4
Needs models/sam_vit_h_4b8939.pth and models/segment_anything.zip (pure Python package, one inode).
Writes output/sam_<split>_<k>of<n>.npz: names, count (segments per image), segs (packed bits, one row per segment)."""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from seg_train import retry
from wmloc.data import LocalisationDataset
from wmloc.utils import downsample_mask

OUT = Path("output")
MIN_CELLS = 6                                    # smaller segments cannot change the score


def load_segments(split):
    """image name -> bool array K x 128 x 128 (all shards of the split)."""
    out = {}
    for f in sorted(OUT.glob(f"sam_{split}_*of*.npz")):
        z = np.load(f)
        segs = np.unpackbits(z["segs"], axis=1).reshape(-1, 128, 128).astype(bool)
        ends = np.cumsum(z["count"])
        for name, a, b in zip(z["names"].tolist(), ends - z["count"], ends):
            out[name] = segs[a:b]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    models = Path(os.environ.get("WMLOC_MODELS", Path(__file__).resolve().parent / "models"))
    sys.path.insert(0, str(models / "segment_anything.zip"))
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=str(models / "sam_vit_h_4b8939.pth")).to("cuda").eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=32, points_per_batch=128)
    ds = LocalisationDataset("release", a.split)
    idx = list(range(a.shard, len(ds), a.of))[:a.limit or None]
    names, count, segs, t0 = [], [], [], time.time()
    for n, i in enumerate(idx):
        it = ds[i]
        found = gen.generate(it["image"])                          # fp32: SAM's post-processing does not take bfloat16
        keep = [downsample_mask(m["segmentation"]) for m in sorted(found, key=lambda m: -m["area"])]
        keep = [k for k in keep if MIN_CELLS <= k.sum() <= k.size - MIN_CELLS]
        names.append(it["image_name"]); count.append(len(keep))
        segs += [np.packbits(k.astype(bool).ravel()) for k in keep]
        if (n + 1) % 20 == 0 or n + 1 == len(idx):
            print(f"{a.split} shard {a.shard}: {n + 1}/{len(idx)} images, {np.mean(count):.0f} segments per image, {time.time() - t0:.0f} s", flush=True)
    f = OUT / f"sam_{a.split}_{a.shard}of{a.of}.npz"
    retry(lambda: np.savez(f, names=np.array(names), count=np.array(count), segs=np.stack(segs)))
    print("wrote", f, flush=True)


if __name__ == "__main__":
    main()
