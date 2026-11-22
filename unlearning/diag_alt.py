"""Does the forgetting transfer to alterations that were never used in training? No training, no checkpoint (round 4).

Half of the forget score is measured on hidden altered copies. Our only local 'aug' view uses the same augment() as
the training. This prints, for the original and for each given checkpoint, accuracy and loss (pseudo-labels = original
top-1 on the clean image, T=1) on all 3500 forget images under views the training never saw, plus how sharp the
outputs are at T=2. Run inside the unlearning data folder:
  python diag_alt.py out/R3a_cf00_s1.25.pt [out/other.pt ...]
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from task_template import DEVICE, MODEL_FILE, load_forget_images, load_model, preprocess
from ulib import augment


def rotate(x, deg):
    a = math.radians(deg)
    theta = torch.tensor([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0]], device=x.device)
    grid = F.affine_grid(theta.expand(x.shape[0], 2, 3), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def center_crop(x, frac):
    h = x.shape[-1]
    m = int(h * (1 - frac) / 2)
    return F.interpolate(x[..., m:h - m, m:h - m], size=h, mode="bilinear", align_corners=False)


def cutout(x, size=80):
    x = x.clone()
    h = x.shape[-1]
    r = (h - size) // 2
    x[..., r:r + size, r:r + size] = 0.0
    return x


def gray(x):
    g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    return g.expand(-1, 3, -1, -1)


VIEWS = {
    "clean": lambda x: x,
    "train_aug": lambda x: augment(x),
    "hflip": lambda x: x.flip(3),
    "blur5": lambda x: F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, stride=1),
    "blur9": lambda x: F.avg_pool2d(F.pad(x, (4, 4, 4, 4), mode="reflect"), 9, stride=1),
    "noise0.1": lambda x: (x + 0.1 * torch.randn_like(x)).clamp(-1, 1),
    "noise0.3": lambda x: (x + 0.3 * torch.randn_like(x)).clamp(-1, 1),
    "gray": gray,
    "down112": lambda x: F.interpolate(F.interpolate(x, size=112, mode="bilinear", align_corners=False), size=224,
                                       mode="bilinear", align_corners=False),
    "down56": lambda x: F.interpolate(F.interpolate(x, size=56, mode="bilinear", align_corners=False), size=224,
                                      mode="bilinear", align_corners=False),
    "shift22": lambda x: torch.roll(x, shifts=(22, 22), dims=(2, 3)),
    "cutout80": cutout,
    "contrast1.5": lambda x: ((x - x.mean(dim=(1, 2, 3), keepdim=True)) * 1.5 + x.mean(dim=(1, 2, 3), keepdim=True)).clamp(-1, 1),
    "bright+0.3": lambda x: (x + 0.3).clamp(-1, 1),
    "dark-0.3": lambda x: (x - 0.3).clamp(-1, 1),
    "rot25": lambda x: rotate(x, 25.0),
    "crop60": lambda x: center_crop(x, 0.6),
}


@torch.no_grad()
def logits(model, imgs, fn, bs=128):
    torch.manual_seed(0)
    out = []
    for i in range(0, len(imgs), bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            out.append(model(fn(preprocess(imgs[i:i + bs]))).float().cpu())
    return torch.cat(out)


def sharp(z):
    p = (z / 2).softmax(1)
    ent = -(p * p.clamp_min(1e-12).log()).sum(1)
    return f"T2 max p {p.max(1).values.mean():.3f}, eff. classes {ent.exp().mean():.1f}"


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    imgs = load_forget_images()
    models = {"original": load_model(MODEL_FILE)}
    for p in paths:
        models[p.name[:28]] = load_model(p)
    y = logits(models["original"], imgs, VIEWS["clean"]).argmax(1)
    print(f"{len(imgs)} forget images; per view: acc / loss mean (std) vs the original's clean top-1")
    print(f"{'view':12s} " + " | ".join(f"{n:>30s}" for n in models))
    for view, fn in VIEWS.items():
        cells = []
        for name, m in models.items():
            z = logits(m, imgs, fn)
            ls = F.cross_entropy(z, y, reduction="none")
            cells.append(f"{(z.argmax(1) == y).float().mean():.3f} / {ls.mean():5.2f} ({ls.std():4.2f})".rjust(30))
        print(f"{view:12s} " + " | ".join(cells), flush=True)
    for name, m in models.items():
        print(f"sharpness on clean forget images, {name}: {sharp(logits(m, imgs, VIEWS['clean']))}")


if __name__ == "__main__":
    main()
