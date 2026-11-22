"""Watermark segmenter: one small fully convolutional network per method, predicting the 128x128 mask from the 512
image. It has to learn the local texture of the watermark, so the input is never downscaled before the network.

Trains on the synthetic set (synth.py) and optionally on real train images, and is judged on real train images with
the official score, so a good number here means it transfers from synthetic to real.
  --real none      synthetic only, judged on ALL real images of the method (tests the generator)
  --real fold:K    synthetic + real images outside fold K, judged on fold K (5 folds, as rect_search)
  --real all       synthetic + all real images, then predicts the test images (for the submission)

  quick local check:  python seg_train.py --method trustmark --synth output/synth_trustmark_quick.npz --epochs 1 --device cpu --limit 16
  on JURECA:          jureca/run.sh watermark watermark/seg_train.py --method trustmark --real none
Writes output/seg_<method>_<real>.pt, output/plots/seg_<method>_<real>_*.png, and for --real all
output/seg_probs_<method>_test.npz (names, probs 128x128)."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rect_search import folds_of
from wmloc.data import LocalisationDataset
from wmloc.utils import miou_image

OUT = Path("output")


# ---------------------------------------------------------------- model
def cbr(cin, cout, stride=1, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, padding=dilation, dilation=dilation, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Residual(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        self.body = nn.Sequential(cbr(c, c, dilation=dilation), nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return F.relu(x + self.body(x))


class Segmenter(nn.Module):
    """512 -> 128 logits. Full resolution stem (the watermark lives in high frequencies), then dilated context."""
    def __init__(self, c=48):
        super().__init__()
        self.net = nn.Sequential(
            cbr(6, c), cbr(c, c),
            cbr(c, 2 * c, stride=2), cbr(2 * c, 2 * c),
            cbr(2 * c, 4 * c, stride=2),
            *[Residual(4 * c, d) for d in (1, 2, 4, 8, 16, 1)],
            nn.Conv2d(4 * c, 1, 1))

    def forward(self, x):                                       # x: B x 3 x 512 x 512 in [0, 1]
        high = x - F.avg_pool2d(x, 5, stride=1, padding=2, count_include_pad=False)
        return self.net(torch.cat([x - 0.5, high * 10], 1))[:, 0]


def loss_fn(logits, target):
    """BCE plus 1 - soft mean IoU over the two classes, the official per-image score made differentiable."""
    p = torch.sigmoid(logits)
    iou = 0
    for pc, tc in ((p, target), (1 - p, 1 - target)):
        inter = (pc * tc).sum((1, 2))
        union = (pc + tc - pc * tc).sum((1, 2))
        iou = iou + (inter + 1) / (union + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + 1 - (iou / 2).mean()


def retry(write, tries=40, wait=30):
    """Run a file write again while the shared filesystem refuses it (inode quota of the whole hackathon)."""
    for k in range(tries):
        try:
            return write()
        except (OSError, RuntimeError) as e:
            print(f"write refused ({type(e).__name__}: {str(e)[:60]}), retry {k + 1}/{tries} in {wait} s", flush=True)
            time.sleep(wait)
    return write()


# ---------------------------------------------------------------- data
def to_batch(images, device):
    return torch.from_numpy(np.ascontiguousarray(images)).to(device).permute(0, 3, 1, 2).float() / 255


def real_set(method, split="train"):
    ds = LocalisationDataset("release", split, methods=[method])
    items = [ds[i] for i in range(len(ds))]
    images = np.stack([it["image"] for it in items])
    masks = np.stack([it["mask128"] for it in items]) if split == "train" else None
    full = np.array([bool(it.get("full_watermark", False)) for it in items])
    return images, masks, full, [it["image_name"] for it in items]


@torch.no_grad()
def predict(model, images, device, batch=16):
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(torch.sigmoid(model(to_batch(images[i:i + batch], device)).float()).cpu().numpy())
    return np.concatenate(out)


def official(probs, masks, thr=0.5):
    return float(np.mean([miou_image((p >= thr).astype(np.uint8), m) for p, m in zip(probs, masks)]))


# ---------------------------------------------------------------- plots
def plots(name, history, images, masks, probs, full):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir = OUT / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ep = [h[0] for h in history]
    ax1.plot(ep, [h[1] for h in history], color="#9aa5b1", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss")
    ax2 = ax1.twinx()
    for k, c, lab in ((2, "#2f80ed", "real score, all"), (3, "#27ae60", "real, type A"), (4, "#eb5757", "real, type B")):
        ax2.plot(ep, [h[k] for h in history], marker="o", color=c, label=lab)
    ax2.axhline(0.45, color="black", ls=":", lw=1, label="current submission (0.45)")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("official score on held-out real images")
    fig.legend(loc="lower right", fontsize=8)
    ax1.set_title(f"Segmenter {name}: learning curve")
    fig.tight_layout()
    fig.savefig(pdir / f"seg_{name}_curve.png", dpi=130)

    idx = list(np.flatnonzero(full)[:3]) + list(np.flatnonzero(~full)[:3])
    fig, axes = plt.subplots(len(idx), 3, figsize=(9, 3 * len(idx)), squeeze=False)
    for row, i in zip(axes, idx):
        row[0].imshow(images[i])
        row[1].imshow(masks[i], vmin=0, vmax=1, cmap="gray")
        row[2].imshow(probs[i], vmin=0, vmax=1, cmap="magma")
        row[0].set_title("type A" if full[i] else "type B", fontsize=8)
        row[1].set_title("true mask", fontsize=8)
        row[2].set_title(f"predicted, score {miou_image((probs[i] >= 0.5).astype(np.uint8), masks[i]):.2f}", fontsize=8)
        for a in row:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(pdir / f"seg_{name}_gallery.png", dpi=100)
    print("plots in", pdir.resolve(), flush=True)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--real", default="none", help="none | fold:K | all")
    ap.add_argument("--synth", default=None, help="default output/synth_<method>.npz")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="quick check: use only N synthetic and N real images")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    print(f"device {a.device}, cuda available {torch.cuda.is_available()}", flush=True)

    z = np.load(a.synth or OUT / f"synth_{a.method}.npz")
    s_img, s_mask = z["images"], z["masks128"]
    r_img, r_mask, r_full, _ = real_set(a.method)
    if a.limit:
        s_img, s_mask, r_img, r_mask, r_full = s_img[:a.limit], s_mask[:a.limit], r_img[:a.limit], r_mask[:a.limit], r_full[:a.limit]
    if a.real == "none":
        tr_img, tr_mask, ev = s_img, s_mask, np.arange(len(r_img))
    else:
        fold = folds_of(np.array([a.method] * len(r_img)), r_full)
        held = int(a.real.split(":")[1]) if a.real.startswith("fold") else -1
        use = np.flatnonzero(fold != held)
        reps = max(1, len(s_img) // (4 * max(len(use), 1)))     # real images about 20 % of each epoch
        tr_img = np.concatenate([s_img] + [r_img[use]] * reps)
        tr_mask = np.concatenate([s_mask] + [r_mask[use]] * reps)
        ev = np.flatnonzero(fold == held) if held >= 0 else np.arange(len(r_img))
    name = f"{a.method}_{a.real.replace(':', '')}"
    print(f"{name}: {len(tr_img)} training images, judged on {len(ev)} real images", flush=True)

    model = Segmenter().to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(tr_img) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    history, t0 = [], time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        order, losses = rng.permutation(len(tr_img)), []
        for i in range(0, len(order), a.batch):
            b = order[i:i + a.batch]
            x, y = to_batch(tr_img[b], a.device), torch.from_numpy(tr_mask[b]).float().to(a.device)
            if rng.random() < 0.5:                                 # flips keep the watermark texture statistics
                x, y = x.flip(-1), y.flip(-1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        probs = predict(model, r_img[ev], a.device)
        f = r_full[ev]
        scores = [official(probs, r_mask[ev])] + [official(probs[sel], r_mask[ev][sel]) if sel.any() else float("nan") for sel in (f, ~f)]
        history.append((epoch, float(np.mean(losses)), *scores))
        print(f"epoch {epoch}/{a.epochs} loss {np.mean(losses):.4f} real score {scores[0]:.4f} "
              f"(type A {scores[1]:.4f}, type B {scores[2]:.4f}), {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), OUT / f"seg_{name}.pt")
    plots(name, history, r_img[ev], r_mask[ev], probs, r_full[ev])

    if a.real == "all":
        t_img, _, _, t_names = real_set(a.method, "test")
        np.savez(OUT / f"seg_probs_{a.method}_test.npz", names=np.array(t_names), probs=predict(model, t_img, a.device).astype(np.float16))
        print("wrote", OUT / f"seg_probs_{a.method}_test.npz", flush=True)


if __name__ == "__main__":
    main()
