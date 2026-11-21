"""Shared helpers for our unlearning methods: augmentations, proxy inputs, soft targets, proxy metrics.

Everything here reuses task_template.py (model builder, server preprocessing) so we never drift from the official interface.
"""
import math

import torch
import torch.nn.functional as F

from task_template import DEVICE, preprocess

T = 2.0  # temperature used by the scorer for every KL


def augment(x: torch.Tensor, scale=(0.75, 1.0), max_rot_deg=10.0, jitter=0.1) -> torch.Tensor:
    """Random flip + resized crop + small rotation + brightness/contrast jitter on preprocessed [-1, 1] images.
    Stand-in for the hidden 'altered copies' of the forget images."""
    b = x.shape[0]
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    s = torch.empty(b, device=x.device).uniform_(*scale)
    a = torch.empty(b, device=x.device).uniform_(-max_rot_deg, max_rot_deg) * math.pi / 180
    tx = (torch.rand(b, device=x.device) * 2 - 1) * (1 - s)
    ty = (torch.rand(b, device=x.device) * 2 - 1) * (1 - s)
    theta = torch.stack([torch.stack([s * a.cos(), -s * a.sin(), tx], 1),
                         torch.stack([s * a.sin(), s * a.cos(), ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    c = torch.empty(b, 1, 1, 1, device=x.device).uniform_(1 - 2 * jitter, 1 + 2 * jitter)
    br = torch.empty(b, 1, 1, 1, device=x.device).uniform_(-jitter, jitter)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - m) * c + m + br).clamp(-1, 1)


def proxy_retain(x: torch.Tensor) -> torch.Tensor:
    """Inputs that are NOT forget images, to keep the student close to the original elsewhere (we have no retain set):
    half mixup (lambda in [0.3, 0.7]) and half cutmix between random pairs of forget images."""
    b = x.shape[0]
    perm = torch.randperm(b, device=x.device)
    lam = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.3, 0.7)
    mix = lam * x + (1 - lam) * x[perm]
    h = x.shape[-1]
    cut = x.clone()
    for i in range(b // 2, b):
        w = int(h * float(torch.empty(1).uniform_(0.4, 0.7)))
        r, c = torch.randint(0, h - w + 1, (2,)).tolist()
        cut[i, :, r:r + w, c:c + w] = x[perm[i], :, r:r + w, c:c + w]
    return torch.cat([mix[: b // 2], cut[b // 2:]])


def soft_target(z: torch.Tensor, target_conf, tau: float = 1.0, other_tau: float = 1.0) -> torch.Tensor:
    """Probabilities at temperature T*tau, with the top class capped at `target_conf` (lower the top logit only).
    The rest of the mass keeps the original's relative ranking, so where the original's runner-up dominates,
    the target flips to it, which is the 'wrong class' a model that never saw the image is most likely to pick."""
    zt = z.float() / (T * tau)
    if other_tau != 1.0:  # flatten the mass among the NON-top classes only (gold = average of several models)
        top = zt.argmax(1, keepdim=True)
        zt = (zt / other_tau).scatter(1, top, zt.gather(1, top))
    c = torch.as_tensor(target_conf, dtype=zt.dtype, device=zt.device).reshape(-1, 1)  # scalar or one cap per image
    if (c < 1).any():
        c = c.clamp(1e-4, 1 - 1e-6)
        top = zt.argmax(1, keepdim=True)
        others = zt.scatter(1, top, float("-inf"))
        lse_others = torch.logsumexp(others, 1, keepdim=True)
        capped = lse_others + torch.log(c / (1 - c))  # top logit giving p_top = c
        zt = zt.scatter(1, top, torch.minimum(zt.gather(1, top), capped))
    return zt.softmax(1)


def _fork_rng():
    """Isolate a fixed-seed evaluation from the caller's RNG stream (CPU + current CUDA device)."""
    return torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else [])


@torch.no_grad()
def difficulty_caps(model, images_u8, conf_min, conf_max, views=4, bs=128, seed=1):
    """Per-image target cap from how robust the original is under STRONG augmentation: the original's mean top-class
    probability (T=2, pseudo-label = clean argmax) over a few strong views, rank-mapped linearly to [conf_min, conf_max].
    Hard images (low robustness) get a low cap, so the loss spreads out like a never-seen model's."""
    model.eval()
    with _fork_rng():
        return _difficulty_caps(model, images_u8, conf_min, conf_max, views, bs, seed)


def _difficulty_caps(model, images_u8, conf_min, conf_max, views, bs, seed):
    torch.manual_seed(seed)
    score = []
    for i in range(0, len(images_u8), bs):
        x = preprocess(images_u8[i:i + bs])
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            y = model(x).argmax(1, keepdim=True)
            p = sum((model(augment(x, scale=(0.4, 0.9), max_rot_deg=25, jitter=0.2)).float() / T).softmax(1).gather(1, y)
                    for _ in range(views)) / views
        score.append(p.squeeze(1).float().cpu())
    score = torch.cat(score)
    rank = score.argsort().argsort().float() / max(len(score) - 1, 1)
    return conf_min + (conf_max - conf_min) * rank


def kl_t(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Per-sample KL(p || q) on softmax(logits / T), as in the scorer."""
    lp = F.log_softmax(p_logits.float() / T, 1)
    lq = F.log_softmax(q_logits.float() / T, 1)
    return (lp.exp() * (lp - lq)).sum(1)


@torch.no_grad()
def logits_views(model, images_u8, views=("clean", "aug", "mix"), bs=128, seed=0):
    """Logits per view, computed like the server (eval mode, fp16 autocast). Same seed => same augmented inputs."""
    model.eval()
    out = {}
    with _fork_rng():
        for v in views:
            out[v] = _view_logits(model, images_u8, v, bs, seed)
    return out


@torch.no_grad()
def _view_logits(model, images_u8, v, bs, seed):
    torch.manual_seed(seed)  # same seed => same augmented inputs for every model compared
    chunks = []
    for i in range(0, len(images_u8), bs):
        x = preprocess(images_u8[i:i + bs])
        if v == "aug":
            x = augment(x)
        elif v == "mix":
            x = proxy_retain(x)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            chunks.append(model(x).float().cpu())
    return torch.cat(chunks)


def proxy_report(orig: dict, sub: dict) -> dict:
    """Proxy metrics on forget images. Pseudo-labels = original argmax on clean images (the original fits its train set).
    clean/aug: accuracy, CE loss mean/std vs pseudo-labels, KL(orig||sub), share of flips that go to the original's
    runner-up. mix: KL(orig||sub) and agreement = retain-drift proxy (lower KL is better)."""
    y = orig["clean"].argmax(1)
    r = {}
    for v in ("clean", "aug"):
        if v not in sub:
            continue
        ls = F.cross_entropy(sub[v], y, reduction="none")
        pred = sub[v].argmax(1)
        flipped = pred != y
        runner = orig[v].topk(2, 1).indices[:, 1]
        r[v] = dict(acc=(pred == y).float().mean().item(), loss_mean=ls.mean().item(), loss_std=ls.std().item(),
                    kl_orig_sub=kl_t(orig[v], sub[v]).mean().item(),
                    flips_to_runner_up=((pred == runner) & flipped).float().sum().item() / max(int(flipped.sum()), 1))
    if "mix" in sub:
        r["mix"] = dict(kl_orig_sub=kl_t(orig["mix"], sub["mix"]).mean().item(),
                        agree=(orig["mix"].argmax(1) == sub["mix"].argmax(1)).float().mean().item())
    return r


def fmt(r: dict) -> str:
    return " | ".join(f"{v}: " + " ".join(f"{k}={x:.4f}" for k, x in m.items()) for v, m in r.items())
