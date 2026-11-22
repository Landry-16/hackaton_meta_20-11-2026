"""Recover the message bits of an existing synthetic set (v3 sets were saved without them) by replaying the random
generator with a dummy embedder: every random draw of the recipe is independent of the watermark model.
Checked against the saved masks. Writes output/<set>_<method>_bits.npy (N x nbits uint8).
  python recover_bits.py --set synthv3 --method trustmark --seed 0"""
import argparse
from pathlib import Path

import numpy as np

import synth
from wmloc.utils import downsample_mask

OUT = Path("output")


class Dummy:
    def embed(self, img, bits):
        return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    z = np.load(OUT / f"{a.set}_{a.method}.npz")
    masks, kinds = z["masks128"], z["kinds"]
    pool, rng = synth.carriers(a.method), np.random.default_rng(a.seed)
    blank = np.zeros((synth.SIZE, synth.SIZE, 3), np.uint8)
    bits = []
    for i in range(len(kinds)):
        rng.choice(len(pool), 2, replace=False)
        _, mask, kind, b = synth.make_sample_v3(Dummy(), a.method, blank, rng)
        assert kind == kinds[i] and np.array_equal(downsample_mask(mask), masks[i]), f"replay diverged at sample {i}"
        bits.append(b)
    np.save(OUT / f"{a.set}_{a.method}_bits.npy", np.array(bits, np.uint8))
    print(f"{a.set} {a.method}: {len(bits)} messages recovered, masks and kinds all match", flush=True)


if __name__ == "__main__":
    main()
