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
    ap.add_argument("--head", default="small", choices=["small", "big"], help="seg_ext head the checkpoints were trained with")
    ap.add_argument("--tta", action="store_true", help="seg_ext models: average with the mirrored image")
    ap.add_argument("--oof", action="store_true", help="instead of the test set: re-predict each held-out fold with its own model, "
                    "write oof_ext_<method>_fold<k>_<tag>_tta.npz (to judge --tta with compare_oof.py)")
    ap.add_argument("--out-prefix", default="seg_probs_ens", help="output/<prefix>_<method>_test.npz (blend.py averages several)")
    ap.add_argument("--scales", default="1", help="seg_ext models trained with --scales")
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
            model = ExtSegmenter(m, load_watermark(m, device="cpu") if m != "stegastamp" else None, head=a.head,
                                 scales=tuple(float(v) for v in a.scales.split(","))).to(a.device)
            run = lambda: predict_ext(model, t_img[s], t_extra[s][:, -1:], a.device, tta=a.tta)
            if a.oof:
                from rect_search import folds_of
                r_img, _, r_methods, r_extra, r_full, r_names = load_split("train")
                sel = np.flatnonzero(r_methods == m)
                fold = folds_of(r_methods[sel], r_full[sel])                     # same split as seg_ext.py
                for k in range(5):
                    model.load_state_dict(torch.load(OUT / f"segext_{m}_fold{k}_{a.tag}.pt", map_location=a.device, weights_only=True))
                    ev = sel[fold == k]
                    probs = predict_ext(model, r_img[ev], r_extra[ev][:, -1:], a.device, tta=True)
                    np.savez(OUT / f"oof_ext_{m}_fold{k}_{a.tag}_tta.npz", names=np.array(r_names)[ev], probs=probs.astype(np.float16))
                    print(f"{m}: fold {k} re-predicted with flip averaging", flush=True)
                continue
        probs = []
        for c in ckpts:
            model.load_state_dict(torch.load(c, map_location=a.device, weights_only=True))
            probs.append(run())
            print(f"{m}: {c.name} predicted", flush=True)
        avg = np.mean(probs, 0).astype(np.float16)
        np.savez(OUT / f"{a.out_prefix}_{m}_test.npz", names=np.array(t_names)[s], probs=avg)
        print(f"{m}: averaged {len(probs)} models -> {a.out_prefix}_{m}_test.npz", flush=True)


if __name__ == "__main__":
    main()
