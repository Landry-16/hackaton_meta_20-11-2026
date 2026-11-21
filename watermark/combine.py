"""Build a test submission from the segmenter probabilities and the rectangle search, optionally submit it.
Per test image:
  - start from the segmenter mask (probability >= threshold, one threshold per method, default 0.5);
  - if the rectangle search found a crop that clearly decodes (its detection threshold, rect_params.json), use that
    rectangle instead: a decoded crop is near certain evidence;
  - without segmenter probabilities for a method, fall back to the rectangle search choice.
  python combine.py                 writes output/submission_combined.pt
  python combine.py --submit        also uploads it (needs HACK_API_KEY)
Inputs from output/: seg_probs_<method>_test.npz, rect_test.npz, rect_params.json."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rect_search import METHODS, box_mask, choose
from wmloc.data import LocalisationDataset

OUT = Path("output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--no-rect", action="store_true", help="segmenter only")
    ap.add_argument("--out", default="submission_combined.pt")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()

    test = LocalisationDataset("release", "test")
    names = [m["image_name"] for m in test.meta]
    methods = {m["image_name"]: m["method"] for m in test.meta}
    seg = {}
    for m in METHODS:
        f = OUT / f"seg_probs_{m}_test.npz"
        if f.exists():
            z = np.load(f)
            seg.update(dict(zip(z["names"].tolist(), z["probs"].astype(np.float32))))
    rect = {}
    if (OUT / "rect_test.npz").exists() and not a.no_rect:
        z, params = np.load(OUT / "rect_test.npz"), json.loads((OUT / "rect_params.json").read_text())
        for n, m, fa, ba, box in zip(z["names"].tolist(), z["methods"], z["full_acc"], z["best_acc"], z["boxes"]):
            rect[n] = (int(choose(np.array([fa]), np.array([ba]), *params[m])[0]), box)

    masks, source = [], {"rectangle": 0, "segmenter": 0, "rect search choice": 0, "zeros": 0}
    for n in names:
        r = rect.get(n)
        if r and r[0] == 1:
            masks.append(box_mask(r[1])); source["rectangle"] += 1
        elif n in seg:
            masks.append((seg[n] >= a.thr).astype(np.uint8)); source["segmenter"] += 1
        elif r:
            masks.append(box_mask(r[1]) if r[0] == 1 else np.full((128, 128), r[0] // 2, np.uint8)); source["rect search choice"] += 1
        else:
            masks.append(np.zeros((128, 128), np.uint8)); source["zeros"] += 1
    masks = np.stack(masks)
    assert masks.shape == (1020, 128, 128) and len(set(names)) == 1020 and set(np.unique(masks)) <= {0, 1}
    path = OUT / a.out
    torch.save({"image_names": names, "masks": torch.from_numpy(masks).bool()}, path)
    print("mask source per image:", source, "| methods with segmenter:", sorted({methods[n] for n in seg}))
    print("wrote", path, f"({path.stat().st_size / 1e6:.1f} MB)")
    if a.submit:
        from submission_template import submit
        submit(str(path))


if __name__ == "__main__":
    main()
