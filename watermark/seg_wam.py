"""Fine-tune the Watermark Anything (WAM, Meta, ICLR 2025) detector to localise OUR three watermarks.
WAM's extractor (SAM ViT-B encoder + pixel decoder) was trained to say, pixel by pixel, where an image carries its own
watermark, under inpainting / splicing style edits: the architecture and pretraining match our task. We keep its
detection channel (0) and fine-tune the whole detector on our real masks + synthetic samples of all three methods.
Needs models/watermark-anything (git clone facebookresearch/watermark-anything) with checkpoints/wam_mit.pth.
  --fold K / --fold all  as seg_real.py (one model for the three methods)
  quick check:  python seg_wam.py --fold 0 --epochs 1 --limit 30 --synth 4 --device cpu
  on JURECA:    jureca/q.sh wam watermark/seg_wam.py --fold 0"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rect_search import METHODS, folds_of
from seg_real import batch_x, load_split, plots, report
from seg_train import loss_fn

OUT = Path("output")
WAM_DIR = Path(os.environ.get("WAM_DIR", "models/watermark-anything"))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_wam_detector():
    sys.path.insert(0, str(WAM_DIR.resolve()))
    from notebooks.inference_utils import load_model_from_checkpoint
    cwd = os.getcwd()
    os.chdir(WAM_DIR)                                  # its configs are relative paths
    try:
        wam = load_model_from_checkpoint("checkpoints/params.json", "checkpoints/wam_mit.pth")
    finally:
        os.chdir(cwd)
    return wam.detector


class WamSegmenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.det = load_wam_detector()

    def forward(self, x):                              # x: B x 3 x 512 x 512 in [0, 1] -> B x 128 x 128 logits
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        return F.avg_pool2d(self.det(x)[:, :1], 2)[:, 0]


@torch.no_grad()
def predict(model, images, device, batch=16):
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(torch.sigmoid(model(batch_x(images[i:i + batch], device)).float()).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4, help="pixel decoder; the ViT encoder gets lr / 10")
    ap.add_argument("--synth", type=int, default=300, help="synthetic samples per method per epoch")
    ap.add_argument("--synth-set", default="synth")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    print(f"device {a.device}, cuda available {torch.cuda.is_available()}", flush=True)
    images, masks, methods, _, full, names = load_split("train", a.limit)
    fold = folds_of(methods, full)
    tr = np.arange(len(images)) if a.fold == "all" else np.flatnonzero(fold != int(a.fold))
    ev = np.arange(len(images)) if a.fold == "all" else np.flatnonzero(fold == int(a.fold))
    synth = {}
    if a.synth:
        for m in METHODS:
            f = OUT / f"{a.synth_set}_{m}.npz"
            if f.exists():
                z = np.load(f)
                synth[m] = (z["images"], z["masks128"])
        print("synthetic sets:", {m: len(v[0]) for m, v in synth.items()}, flush=True)
    name = f"wam_fold{a.fold}" + (f"_{a.synth_set}{a.synth}" if a.synth else "")
    print(f"{name}: train {len(tr)} real images, judged on {len(ev)}", flush=True)

    model = WamSegmenter().to(a.device)
    enc = list(model.det.image_encoder.parameters())
    dec = list(model.det.pixel_decoder.parameters())
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr / 10}, {"params": dec, "lr": a.lr}], weight_decay=1e-4)
    per_epoch = len(tr) + a.synth * len(synth)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr / 10, a.lr], total_steps=a.epochs * ((per_epoch + a.batch - 1) // a.batch), pct_start=0.1)
    history, t0, probs = [], time.time(), None
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = [("real", None, i) for i in rng.permutation(tr)]
        for m, (si, _) in synth.items():
            order += [("synth", m, i) for i in rng.choice(len(si), min(a.synth, len(si)), replace=False)]
        order = [order[k] for k in rng.permutation(len(order))]
        losses = []
        for i in range(0, len(order), a.batch):
            chunk = order[i:i + a.batch]
            x = batch_x(np.stack([images[j] if s == "real" else synth[m][0][j] for s, m, j in chunk]), a.device)
            y = torch.from_numpy(np.stack([masks[j] if s == "real" else synth[m][1][j] for s, m, j in chunk])).float().to(a.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        if a.fold != "all" and (epoch % 5 == 0 or epoch == a.epochs):
            probs = predict(model, images[ev], a.device)
            overall, rows = report(probs, masks[ev], methods[ev], full[ev])
            history.append((epoch, overall, rows))
            print(f"epoch {epoch}/{a.epochs} loss {np.mean(losses):.4f} held-out {overall:.4f} | " +
                  " ".join(f"{m} {v[0]:.3f} (A {v[1]:.3f} B {v[2]:.3f})" for m, v in rows.items()) + f" | {time.time() - t0:.0f} s", flush=True)
        else:
            print(f"epoch {epoch}/{a.epochs} loss {np.mean(losses):.4f} | {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), OUT / f"seg{name}.pt")
    if a.fold != "all":
        plots(name, history, images[ev], masks[ev], probs, methods[ev], full[ev])
        np.savez(OUT / f"oof_{name}.npz", names=np.array(names)[ev], probs=probs.astype(np.float16))
        return
    t_img, _, t_methods, _, _, t_names = load_split("test")
    probs = predict(model, t_img, a.device).astype(np.float16)
    for m in METHODS:
        s = t_methods == m
        np.savez(OUT / f"seg_probs_wam_{m}_test.npz", names=np.array(t_names)[s], probs=probs[s])
    print("wrote seg_probs_wam_<method>_test.npz", flush=True)


if __name__ == "__main__":
    main()
