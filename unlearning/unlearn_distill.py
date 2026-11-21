"""Method A: soft-target distillation on the last blocks, with rewind checkpoints.

On (augmented) forget images, the student is pulled towards a softened version of the original's prediction
(top class capped at --target-conf, runner-up ranking kept). On proxy retain inputs (mixup/cutmix of forget images),
it is pulled towards the original (KL at T=2). Only the last --train-blocks blocks + final norm + head move.
A float32 state dict is saved after every epoch (rewind: pick the epoch / setting by public score).

Run inside the unlearning data folder (jureca/job.sh does it):
  sbatch -J unl-distill jureca/job.sh unlearning unlearning/unlearn_distill.py --target-conf 0.5 --epochs 4
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from task_template import DEVICE, MODEL_FILE, load_forget_images, load_model, preprocess
from ulib import T, augment, fmt, logits_views, proxy_report, proxy_retain, soft_target


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--target-conf", type=float, default=0.5, help="cap on the target top-class prob (at T=2)")
    p.add_argument("--tau", type=float, default=1.0, help="extra temperature on the target (>1 = softer)")
    p.add_argument("--alpha", type=float, default=1.0, help="weight of the proxy-retain KL to the original")
    p.add_argument("--l2sp", type=float, default=0.0, help="weight of ||w - w_orig||^2 on trained params")
    p.add_argument("--train-blocks", type=int, default=4, help="number of last transformer blocks to train")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--n", type=int, default=0, help="use only the first n forget images (0 = all; debug)")
    p.add_argument("--eval-n", type=int, default=1000, help="images used for the per-epoch proxy report")
    p.add_argument("--out", default="out/distill", help="checkpoint prefix, relative to the data folder")
    p.add_argument("--clean-frac", type=float, default=0.5, help="share of each forget batch left un-augmented")
    p.add_argument("--save-every", type=int, default=1, help="save + report every N epochs (and at the last one)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def trainable(model, k):
    names = [f"blocks.{i}." for i in range(len(model.blocks) - k, len(model.blocks))] + ["norm.", "fc_norm.", "head."]
    params = []
    for n, p in model.named_parameters():
        p.requires_grad = any(n.startswith(x) for x in names)
        if p.requires_grad:
            params.append(p)
    return params


def main():
    a = parse()
    torch.manual_seed(a.seed)
    tag = f"tc{a.target_conf}_tau{a.tau}_a{a.alpha}_k{a.train_blocks}_lr{a.lr}_cf{a.clean_frac}"
    out = Path(a.out + "_" + tag)
    out.parent.mkdir(parents=True, exist_ok=True)

    imgs = load_forget_images()
    if a.n:
        imgs = imgs[: a.n]
    teacher = load_model(MODEL_FILE).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = load_model(MODEL_FILE)
    params = trainable(student, a.train_blocks)
    anchor = [p.detach().clone() for p in params]
    print(f"{tag}: {len(imgs)} images, {sum(p.numel() for p in params) / 1e6:.1f}M trainable params, device {DEVICE}")

    ev = imgs[: a.eval_n]
    orig_views = logits_views(teacher, ev)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = a.epochs * ((len(imgs) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda")

    for ep in range(1, a.epochs + 1):
        student.train()
        t0, tot = time.time(), [0.0, 0.0]
        for idx in torch.randperm(len(imgs)).split(a.bs):
            x = preprocess(imgs[idx])
            keep = torch.rand(len(x), device=x.device) < a.clean_frac
            xf = torch.where(keep[:, None, None, None], x, augment(x))
            xr = proxy_retain(augment(x))
            with torch.no_grad(), torch.autocast(**amp):
                zf, zr = teacher(xf).float(), teacher(xr).float()
            with torch.autocast(**amp):
                sf, sr = student(xf).float(), student(xr).float()
            target = soft_target(zf, a.target_conf, a.tau)
            lf = F.kl_div(F.log_softmax(sf / T, 1), target, reduction="batchmean")
            lr_ = F.kl_div(F.log_softmax(sr / T, 1), F.softmax(zr / T, 1), reduction="batchmean")
            loss = lf + a.alpha * lr_
            if a.l2sp:
                loss = loss + a.l2sp * sum(((p - p0) ** 2).sum() for p, p0 in zip(params, anchor))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot[0] += lf.item() * len(idx)
            tot[1] += lr_.item() * len(idx)
        if ep % a.save_every and ep != a.epochs:
            print(f"epoch {ep} ({time.time() - t0:.0f}s) forget_kl={tot[0] / len(imgs):.4f} retain_kl={tot[1] / len(imgs):.4f}", flush=True)
            continue
        sd = {k: v.detach().cpu().float() for k, v in student.state_dict().items()}
        assert all(torch.isfinite(v).all() for v in sd.values()), "NaN/Inf in weights"
        path = Path(f"{out}_e{ep}.pt")
        torch.save(sd, path)
        rep = proxy_report(orig_views, logits_views(student, ev))
        print(f"epoch {ep} ({time.time() - t0:.0f}s) forget_kl={tot[0] / len(imgs):.4f} retain_kl={tot[1] / len(imgs):.4f}"
              f" -> {path}\n  {fmt(rep)}", flush=True)


if __name__ == "__main__":
    main()
