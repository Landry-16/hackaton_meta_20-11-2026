"""Segmenter trained on the REAL train masks only (synthetic data did not transfer).
One model for all three methods: a pretrained ResNet-34 encoder (ImageNet) and a light decoder to the 128 grid.
Extra input planes, broadcast to the decoder: the method (one-hot) and the full-image bit accuracy from the decode
(tells type A from type B, the costliest mistake).
  --fold K     train on the 4 other folds, judge on fold K (same 5 folds as rect_search)
  --fold all   train on all 510 images, predict the test set (writes seg_probs_<method>_test.npz for combine.py)
  quick local check:  python seg_real.py --fold 0 --epochs 1 --limit 24 --device cpu
  on JURECA:          GPU=1 jureca/run.sh watermark watermark/seg_real.py --fold 0
The full-image bit accuracy comes from output/rect_train.npz and output/rect_test.npz (rect_search.py)."""
import argparse
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from rect_search import METHODS, folds_of, load_cache
from seg_train import loss_fn, official
from wmloc.data import LocalisationDataset

OUT = Path("output")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_synth(sets, method):
    """One or several synthetic sets ("synthv3" or "synthv3,synthv3b"), concatenated."""
    parts = [np.load(OUT / f"{s}_{method}.npz") for s in sets.split(",")]
    out = {"images": np.concatenate([p["images"] for p in parts]), "masks128": np.concatenate([p["masks128"] for p in parts]),
           "kinds": np.concatenate([p["kinds"] for p in parts])}
    if all("acc" in p.files for p in parts):
        out["acc"] = np.concatenate([p["acc"] for p in parts])
    return out


class RealSegmenter(nn.Module):
    def __init__(self, pretrained=True, extra=len(METHODS) + 1):
        super().__init__()
        self.enc = timm.create_model("resnet34", pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        chans = self.enc.feature_info.channels()                  # strides 4, 8, 16, 32
        self.lat = nn.ModuleList(nn.Conv2d(c, 64, 1) for c in chans)
        self.head = nn.Sequential(nn.Conv2d(64 * 4 + extra, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 1, 1))

    def forward(self, x, extra):                                 # x: B x 3 x 512 x 512 in [0, 1]; extra: B x E
        feats = self.enc((x - MEAN.to(x.device)) / STD.to(x.device))
        size = feats[0].shape[-2:]                               # 128 x 128
        ups = [F.interpolate(l(f), size=size, mode="bilinear", align_corners=False) for l, f in zip(self.lat, feats)]
        planes = extra[:, :, None, None].expand(-1, -1, *size)
        return self.head(torch.cat(ups + [planes], 1))[:, 0]


def load_split(split, limit=0):
    ds = LocalisationDataset("release", split)
    n = min(limit or len(ds), len(ds))
    items = [ds[i] for i in range(n)]
    cache = load_cache(split)
    acc = dict(zip(cache["names"].tolist(), cache["full_acc"]))
    images = np.stack([it["image"] for it in items])
    methods = np.array([it["method"] for it in items])
    extra = np.stack([[float(it["method"] == m) for m in METHODS] + [acc.get(it["image_name"], 0.5)] for it in items]).astype(np.float32)
    masks = np.stack([it["mask128"] for it in items]) if split == "train" else None
    full = np.array([bool(it.get("full_watermark", False)) for it in items])
    return images, masks, methods, extra, full, [it["image_name"] for it in items]


def batch_x(images, device):
    return torch.from_numpy(np.ascontiguousarray(images)).to(device).permute(0, 3, 1, 2).float() / 255


@torch.no_grad()
def predict(model, images, extra, device, batch=16, tta=True):
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        x, e = batch_x(images[i:i + batch], device), torch.from_numpy(extra[i:i + batch]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            p = torch.sigmoid(model(x, e).float())
            if tta:                                                   # flip TTA
                p = (p + torch.sigmoid(model(x.flip(-1), e).float()).flip(-1)) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def report(probs, masks, methods, full):
    rows = {}
    for m in METHODS:
        s = methods == m
        rows[m] = (official(probs[s], masks[s]), official(probs[s & full], masks[s & full]) if (s & full).any() else np.nan,
                   official(probs[s & ~full], masks[s & ~full]) if (s & ~full).any() else np.nan)
    return official(probs, masks), rows


def plots(name, history, images, masks, probs, methods, full):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir = OUT / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ep = [h[0] for h in history]
    ax.plot(ep, [h[1] for h in history], color="black", lw=2, label="all methods")
    for m, c in zip(METHODS, ("#eb5757", "#27ae60", "#9b51e0")):
        ax.plot(ep, [h[2][m][0] for h in history], color=c, label=m)
    ax.axhline(0.45, color="gray", ls=":", label="our submission (0.45)")
    ax.axhline(0.70, color="#f2994a", ls="--", label="leaderboard top (0.70)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("official score on the held-out fold")
    ax.set_title(f"Real-data segmenter {name}")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(pdir / f"segreal_{name}_curve.png", dpi=130)
    idx = [i for m in METHODS for i in (list(np.flatnonzero((methods == m) & full)[:1]) + list(np.flatnonzero((methods == m) & ~full)[:1]))]
    fig, axes = plt.subplots(len(idx), 3, figsize=(9, 3 * len(idx)), squeeze=False)
    from wmloc.utils import miou_image
    for row, i in zip(axes, idx):
        row[0].imshow(images[i]); row[1].imshow(masks[i], vmin=0, vmax=1, cmap="gray"); row[2].imshow(probs[i], vmin=0, vmax=1, cmap="magma")
        row[0].set_title(f"{methods[i]}, type {'A' if full[i] else 'B'}", fontsize=8)
        row[1].set_title("true mask", fontsize=8)
        row[2].set_title(f"predicted, score {miou_image((probs[i] >= 0.5).astype(np.uint8), masks[i]):.2f}", fontsize=8)
        for a in row:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(pdir / f"segreal_{name}_gallery.png", dpi=100)
    print("plots in", pdir.resolve(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0", help="0..4 or all")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--synth", type=int, default=0, help="synthetic samples per method per epoch")
    ap.add_argument("--synth-set", default="synthv2")
    ap.add_argument("--no-real", action="store_true", help="train on synthetic samples only (transfer gate), judge on ALL real images")
    ap.add_argument("--no-flip", action="store_true", help="no flip augmentation or TTA (StegaStamp decoding dies under flips)")
    ap.add_argument("--save-methods", default=",".join(METHODS), help="with --fold all: methods whose test probabilities to write")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    print(f"device {a.device}, cuda available {torch.cuda.is_available()}", flush=True)
    images, masks, methods, extra, full, _names = load_split("train", a.limit)
    fold = folds_of(methods, full)
    tr = np.arange(len(images)) if a.fold == "all" else np.flatnonzero(fold != int(a.fold))
    ev = np.arange(len(images)) if a.fold == "all" else np.flatnonzero(fold == int(a.fold))
    if a.no_real:
        tr, ev = np.array([], int), np.arange(len(images))
    name = f"fold{a.fold}" + ("_synthonly" if a.no_real else "")
    print(f"{name}: train {len(tr)} real images, judged on {len(ev)}", flush=True)

    model = RealSegmenter(pretrained=not a.limit).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    synth = {}
    if a.synth:
        for k, m in enumerate(METHODS):
            z = load_synth(a.synth_set, m)
            onehot = np.zeros(len(METHODS) + 1, np.float32); onehot[k] = 1
            ex = np.stack([onehot] * len(z["kinds"])); ex[:, -1] = z["acc"] if "acc" in z else np.where(z["kinds"] == "full", 0.97, 0.53)
            synth[m] = (z["images"], z["masks128"], ex)
        name += f"_{a.synth_set}{a.synth}"
    steps = a.epochs * ((len(tr) + a.synth * len(synth) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    history, t0 = [], time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = [(None, i) for i in rng.permutation(tr)]
        for m, (si, _, _) in synth.items():
            order += [(m, i) for i in rng.choice(len(si), a.synth, replace=False)]
        order = [order[k] for k in rng.permutation(len(order))]
        losses = []
        for i in range(0, len(order), a.batch):
            chunk = order[i:i + a.batch]
            get = lambda real, j: np.stack([real[q] if m is None else synth[m][j][q] for m, q in chunk])
            x = batch_x(get(images, 0), a.device)
            y = torch.from_numpy(get(masks, 1)).float().to(a.device)
            e = torch.from_numpy(get(extra, 2)).to(a.device)
            if not a.no_flip and rng.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if not a.no_flip and rng.random() < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
                loss = loss_fn(model(x, e).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        if a.fold != "all" and (epoch % 4 == 0 or epoch == a.epochs):
            probs = predict(model, images[ev], extra[ev], a.device, tta=not a.no_flip)
            overall, rows = report(probs, masks[ev], methods[ev], full[ev])
            history.append((epoch, overall, rows))
            print(f"epoch {epoch}/{a.epochs} loss {np.mean(losses):.4f} held-out {overall:.4f} | " +
                  " ".join(f"{m} {v[0]:.3f} (A {v[1]:.3f} B {v[2]:.3f})" for m, v in rows.items()) + f" | {time.time() - t0:.0f} s", flush=True)
        else:
            print(f"epoch {epoch}/{a.epochs} loss {np.mean(losses):.4f} | {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), OUT / f"segreal_{name}.pt")
    if a.fold != "all":
        plots(name, history, images[ev], masks[ev], probs, methods[ev], full[ev])
        np.savez(OUT / f"oof_real_{name}{'_noflip' if a.no_flip else ''}.npz", names=np.array(_names)[ev], probs=probs.astype(np.float16))
        return
    t_img, _, t_methods, t_extra, _, t_names = load_split("test")
    probs = predict(model, t_img, t_extra, a.device, tta=not a.no_flip).astype(np.float16)
    for m in a.save_methods.split(","):
        s = t_methods == m
        np.savez(OUT / f"seg_probs_{m}_test.npz", names=np.array(t_names)[s], probs=probs[s])
    print("wrote seg_probs_<method>_test.npz for", a.save_methods, flush=True)


if __name__ == "__main__":
    main()
