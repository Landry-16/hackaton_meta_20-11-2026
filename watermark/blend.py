"""Average the probabilities of several seg_ext variants of one method (different synthetic sets, volumes, scales):
independent fits make independent errors. Nothing is fitted: plain mean, judged with compare_oof.py like any variant.
  python blend.py --method trustmark --name blendA --sources "synthv3,synthv3b2000_long;synthv5,synthv5b2000_long"
      -> output/oof_ext_<method>_fold<k>_<name>.npz   (then compare_oof.py --a <best single> --b <name>)
  python blend.py --method trustmark --name blendA --test "seg_probs_long;seg_probs_v5"
      -> output/seg_probs_ens_<method>_test.npz from output/<prefix>_<method>_test.npz files (ens_predict.py --out-prefix)"""
import argparse
from pathlib import Path

import numpy as np

OUT = Path("output")


def mean_by_name(files):
    acc, names = {}, None
    for f in files:
        z = np.load(f)
        names = z["names"].tolist() if names is None else names
        for n, p in zip(z["names"].tolist(), z["probs"].astype(np.float32)):
            acc.setdefault(n, []).append(p)
    assert all(len(v) == len(files) for v in acc.values()), "the variants do not cover the same images"
    return np.array(names), np.stack([np.mean(acc[n], 0) for n in names]).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--sources", default="", help="OOF sources separated by ;")
    ap.add_argument("--test", default="", help="test probability prefixes separated by ;")
    a = ap.parse_args()
    if a.sources:                                               # any OOF source postproc.py knows (seg_ext or real_...)
        from postproc import oof_probs
        parts = [oof_probs(a.method, src) for src in a.sources.split(";")]
        from wmloc.data import LocalisationDataset
        own = {m["image_name"] for m in LocalisationDataset("release", "train").meta if m["method"] == a.method}
        names = sorted(set.intersection(own, *(set(d) for d in parts)))          # real_ sources hold the 3 methods
        assert len(names) == 170, f"{len(names)} images are covered by all the variants, 170 expected"
        probs = np.stack([np.mean([d[n] for d in parts], 0) for n in names]).astype(np.float16)
        for k in range(1, 5):
            (OUT / f"oof_ext_{a.method}_fold{k}_{a.name}.npz").unlink(missing_ok=True)
        np.savez(OUT / f"oof_ext_{a.method}_fold0_{a.name}.npz", names=np.array(names), probs=probs)   # one file holds the 5 folds
        print(f"{a.method}: {a.name} = mean of {len(parts)} variants, {len(names)} images written")
    if a.test:
        names, probs = mean_by_name([OUT / f"{p}_{a.method}_test.npz" for p in a.test.split(";")])
        np.savez(OUT / f"seg_probs_ens_{a.method}_test.npz", names=names, probs=probs)
        print(f"{a.method}: test probabilities = mean of {len(a.test.split(';'))} variants")


if __name__ == "__main__":
    main()
