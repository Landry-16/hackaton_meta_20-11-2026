"""Decide if variant B replaces variant A for one method, from out-of-fold predictions on the 170 train images.
Paired per-image difference of the official score at threshold 0.5 (no tuning): B is adopted only if the mean gain
is above both 0.005 and 2 standard errors. Needs the 5 folds of both variants.
  python compare_oof.py --method pixelseal --a synthv2400 --b "synthv3,synthv3b600"
Sources as in postproc.py: "<synth-set><n>" for seg_ext, "real_<suffix>" for seg_real."""
import argparse

import numpy as np

from postproc import oof_probs
from wmloc.data import LocalisationDataset
from wmloc.utils import miou_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--a", required=True, help="current best")
    ap.add_argument("--b", required=True, help="candidate")
    a = ap.parse_args()
    ds = LocalisationDataset("release", "train")
    meta = {m["image_name"]: m for m in ds.meta if m["method"] == a.method}
    pa, pb = oof_probs(a.method, a.a), oof_probs(a.method, a.b)
    names = sorted(n for n in meta if n in pa and n in pb)
    if len(names) < len(meta):
        print(f"INCOMPLETE: {len(names)}/{len(meta)} images have both variants (all 5 folds needed), no decision")
    if not names:
        return
    sc = lambda p: np.array([miou_image((p[n] >= 0.5).astype(np.uint8), ds.m128[n]) for n in names])
    sa, sb = sc(pa), sc(pb)
    full = np.array([bool(meta[n]["full_watermark"]) for n in names])
    d = sb - sa
    se = d.std() / np.sqrt(len(d))
    print(f"{a.method} n={len(names)} | A {a.a}: {sa.mean():.4f} (type A {sa[full].mean():.3f}, B {sa[~full].mean():.3f}) | "
          f"B {a.b}: {sb.mean():.4f} (type A {sb[full].mean():.3f}, B {sb[~full].mean():.3f})")
    ok = len(names) == len(meta) and d.mean() > max(0.005, 2 * se)
    print(f"paired gain {d.mean():+.4f} +- {se:.4f} -> {'ADOPT B' if ok else 'KEEP A'}")


if __name__ == "__main__":
    main()
