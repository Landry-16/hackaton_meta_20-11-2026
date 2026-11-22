"""Synthesise an in-distribution retain anchor by model inversion of the ORIGINAL model (DeepInversion-style, ViT).

Images are optimised from noise (or from external photos) so that the original assigns them a chosen class, with
priors: random jitter + flip, total variation, l2, and a feature-statistics term that matches the per-block token
mean / std of the generated batch to the statistics of real in-distribution images (aggregated over the forget set,
a ViT stand-in for DeepInversion's BatchNorm statistics). Output: one npz, key "images", uint8 [N, 224, 224, 3],
same layout as external_images.npz, usable with unlearn_distill.py --external <file> --holdout <small>.

Run inside the unlearning data folder, on a GPU (team queue):
  python invert.py --classes forget --per-class 40 --out external_inv.npz       # the 45 forget classes first
  python invert.py --classes rest   --per-class 40 --out external_inv_rest.npz  # the other 55
  python invert.py --merge external_inv.npz external_inv_rest.npz --out external_inv.npz
File-count rule: --out may be an EXISTING file (it is overwritten in place, no new inode); e.g. first
  mv out/<obsolete checkpoint>.pt external_inv.npz
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from task_template import DEVICE, MODEL_FILE, load_forget_images, load_model, preprocess

T = 2.0


def save_npz(path: Path, images: np.ndarray) -> None:
    """Write into the existing file if there is one (no new inode on a full file system)."""
    mode = "r+b" if path.exists() else "wb"
    with path.open(mode) as f:
        f.truncate(0)
        f.seek(0)
        np.savez(f, images=images)


def block_stats(model, x):
    """Per-block token mean / std (over batch and tokens), via hooks on every transformer block output."""
    feats = []
    hooks = [b.register_forward_hook(lambda m, i, o: feats.append(o)) for b in model.blocks]
    logits = model(x)
    for h in hooks:
        h.remove()
    stats = [(f.float().mean(dim=(0, 1)), f.float().std(dim=(0, 1))) for f in feats]
    return logits, stats


def tv_loss(x):
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


def jitter_flip(x, max_shift):
    if max_shift:
        dx, dy = torch.randint(-max_shift, max_shift + 1, (2,)).tolist()
        x = torch.roll(x, shifts=(dx, dy), dims=(2, 3))
    if torch.rand(1).item() < 0.5:
        x = x.flip(3)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classes", default="forget", help="forget | rest | all | comma-separated class ids")
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--w-tv", type=float, default=0.05)
    p.add_argument("--w-l2", type=float, default=1e-4)
    p.add_argument("--w-feat", type=float, default=0.1, help="weight of the block-statistics matching term")
    p.add_argument("--n-ref", type=int, default=512, help="real forget images used for the reference statistics")
    p.add_argument("--jitter", type=int, default=16)
    p.add_argument("--init", default="noise", help="noise | path to an npz of photos to start from")
    p.add_argument("--out", type=Path, default=Path("external_inv.npz"))
    p.add_argument("--merge", type=Path, nargs="+", default=None, help="concatenate these npz files into --out")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if a.merge:
        imgs = np.concatenate([np.load(f)["images"] for f in a.merge])
        save_npz(a.out, imgs)
        print(f"merged {len(a.merge)} files -> {a.out}: {imgs.shape}")
        return

    torch.manual_seed(a.seed)
    model = load_model(MODEL_FILE).eval()
    for q in model.parameters():
        q.requires_grad = False
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda")

    forget = load_forget_images()
    with torch.no_grad():
        pseudo = torch.cat([model(preprocess(forget[i:i + 128])).argmax(1).cpu() for i in range(0, len(forget), 128)])
        ref_idx = torch.randperm(len(forget))[: a.n_ref]
        ref_means, ref_stds, nb = None, None, 0
        for i in range(0, len(ref_idx), 64):
            with torch.autocast(**amp):
                _, st = block_stats(model, preprocess(forget[ref_idx[i:i + 64]]))
            m = [s[0] for s in st]
            sd = [s[1] for s in st]
            ref_means = m if ref_means is None else [r + x for r, x in zip(ref_means, m)]
            ref_stds = sd if ref_stds is None else [r + x for r, x in zip(ref_stds, sd)]
            nb += 1
        ref_means = [r / nb for r in ref_means]
        ref_stds = [r / nb for r in ref_stds]
    forget_classes = sorted(set(pseudo.tolist()))
    if a.classes == "forget":
        classes = forget_classes
    elif a.classes == "rest":
        classes = [c for c in range(100) if c not in forget_classes]
    elif a.classes == "all":
        classes = forget_classes + [c for c in range(100) if c not in forget_classes]
    else:
        classes = [int(c) for c in a.classes.split(",")]
    labels = torch.tensor([c for c in classes for _ in range(a.per_class)])
    print(f"{len(classes)} classes x {a.per_class} = {len(labels)} images, {a.iters} iters per batch of {a.bs}",
          flush=True)

    init_pool = None
    if a.init != "noise":
        init_pool = torch.from_numpy(np.load(a.init)["images"])
    out = np.zeros((len(labels), 224, 224, 3), dtype=np.uint8)
    t0 = time.time()
    for b0 in range(0, len(labels), a.bs):
        y = labels[b0:b0 + a.bs].to(DEVICE)
        if init_pool is None:
            x = (torch.randn(len(y), 3, 224, 224, device=DEVICE) * 0.2).requires_grad_(True)
        else:
            x = preprocess(init_pool[torch.randint(len(init_pool), (len(y),))]).clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=a.lr, betas=(0.5, 0.9))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
        for it in range(a.iters):
            xj = jitter_flip(x, a.jitter)
            with torch.autocast(**amp):
                logits, st = block_stats(model, xj)
            ce = F.cross_entropy(logits.float(), y)
            feat = sum((m - rm).norm() + (s - rs).norm() for (m, s), rm, rs in zip(st, ref_means, ref_stds))
            loss = ce + a.w_tv * tv_loss(x) + a.w_l2 * x.pow(2).mean() + a.w_feat * feat
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                x.clamp_(-1, 1)
        img = ((x.detach() * 0.5 + 0.5).clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        out[b0:b0 + len(y)] = img
        print(f"  batch {b0 // a.bs + 1}/{(len(labels) + a.bs - 1) // a.bs}: ce {ce.item():.3f} feat {feat.item():.2f}"
              f" ({time.time() - t0:.0f}s)", flush=True)
        save_npz(a.out, out[: b0 + len(y)])  # partial result survives a crash or time limit

    # sanity: the original on the saved uint8 images, exactly as --external will feed them
    with torch.no_grad():
        u8 = torch.from_numpy(out)
        probs = torch.cat([F.softmax(model(preprocess(u8[i:i + 128])).float(), 1).cpu() for i in range(0, len(u8), 128)])
    top1 = probs.argmax(1)
    conf = probs.gather(1, labels[:, None]).squeeze(1)
    print(f"original on the saved images: hit rate {(top1 == labels).float().mean():.3f}, "
          f"mean p(target) {conf.mean():.3f}")
    for c in classes[:10]:
        m = labels == c
        print(f"  class {c}: hit {(top1[m] == c).float().mean():.2f}, p {conf[m].mean():.2f}")
    print(f"wrote {a.out}: {out.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
