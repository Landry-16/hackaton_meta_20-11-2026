"""Average the test predictions of already trained models: the 5 fold models (+ the model trained on all images).
Averaging independent fits reduces variance; nothing is tuned on the test set.
  python ens_predict.py --tag synthv2400 --methods pixelseal,trustmark          (seg_ext checkpoints)
  python ens_predict.py --real-tag synthv2300 --methods stegastamp              (seg_real checkpoints, flip TTA)
Writes output/seg_probs_ens_<method>_test.npz, read by postproc.py --test-prefix seg_probs_ens."""
import argparse
from pathlib import Path

import numpy as np
import torch

from seg_real import RealSegmenter, load_split
from seg_real import predict as predict_real

OUT = Path("output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="pixelseal,trustmark")
    ap.add_argument("--tag", default="synthv2400", help="seg_ext checkpoint suffix")
    ap.add_argument("--real-tag", default=None, help="seg_real checkpoint suffix (then methods come from seg_real)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    t_img, _, t_methods, t_extra, _, t_names = load_split("test")
    for m in a.methods.split(","):
        s = t_methods == m
        if a.real_tag:
            ckpts = sorted(OUT.glob(f"segreal_fold*_{a.real_tag}.pt"))
            model = RealSegmenter(pretrained=False).to(a.device)
            run = lambda: predict_real(model, t_img[s], t_extra[s], a.device, tta=True)
        else:
            from seg_ext import ExtSegmenter, predict as predict_ext
            from wmloc.watermarks import load_watermark
            ckpts = sorted(OUT.glob(f"segext_{m}_fold*_{a.tag}.pt"))
            model = ExtSegmenter(m, load_watermark(m, device="cpu") if m != "stegastamp" else None).to(a.device)
            run = lambda: predict_ext(model, t_img[s], t_extra[s][:, -1:], a.device)
        probs = []
        for c in ckpts:
            model.load_state_dict(torch.load(c, map_location=a.device, weights_only=True))
            probs.append(run())
            print(f"{m}: {c.name} predicted", flush=True)
        avg = np.mean(probs, 0).astype(np.float16)
        np.savez(OUT / f"seg_probs_ens_{m}_test.npz", names=np.array(t_names)[s], probs=avg)
        print(f"{m}: averaged {len(probs)} models -> seg_probs_ens_{m}_test.npz", flush=True)


if __name__ == "__main__":
    main()
