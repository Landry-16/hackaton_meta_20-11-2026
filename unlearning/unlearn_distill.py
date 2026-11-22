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

from task_template import DEVICE, MODEL_FILE, build_model, load_forget_images, load_model, preprocess
from ulib import (T, augment, class_wrong_dist, count_caps, difficulty_caps, fmt, kl_t, logits_views, mix_class_target,
                  proxy_report, proxy_retain, reserve, soft_target)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--target-conf", type=float, default=0.5, help="cap on the target top-class prob (at T=2)")
    p.add_argument("--conf-min", type=float, default=None, help="per-image caps: hardest image gets conf-min ...")
    p.add_argument("--conf-max", type=float, default=None, help="... easiest gets conf-max (overrides --target-conf)")
    p.add_argument("--cap-by", choices=["rank", "count"], default="rank",
                   help="per-image caps from difficulty rank (default) or one cap per pseudo-class from its number of "
                        "forget images (--conf-min for the fullest classes, --conf-max for the smallest, geometric)")
    p.add_argument("--cap-spacing", choices=["linear", "log"], default="linear",
                   help="per-image caps spread linearly (default) or geometrically between conf-min and conf-max")
    p.add_argument("--class-mix", type=float, default=0.0,
                   help="w in [0,1]: blend the wrong-class mass with the class average of the original's non-top "
                        "distribution (top-class mass unchanged)")
    p.add_argument("--other-tau", type=float, default=1.0,
                   help=">1 spreads the target mass over the wrong classes (top-class cap unchanged)")
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
    p.add_argument("--external", default=None, help="npz of unlabeled external images (prepare_external.py) used as "
                   "retain anchor instead of mixup/cutmix of forget images")
    p.add_argument("--holdout", type=int, default=1000, help="last N external images: not trained on, used for drift")
    p.add_argument("--forget-teacher", default=None,
                   help="checkpoint used as teacher on the forget images only: target = softmax(forget_scale * M(x) / T),"
                        " no cap / other-tau / class-mix; the anchor term still distills the original model")
    p.add_argument("--forget-scale", type=float, default=1.0, help="logit scale applied to --forget-teacher")
    p.add_argument("--forget-weight", type=float, default=1.0,
                   help="weight of the forget term; 0 = anchor-only training, the forget images are never used")
    p.add_argument("--reinit-blocks", type=int, default=0,
                   help="re-initialise the last K transformer blocks with a freshly built model before training")
    p.add_argument("--reinit-head", action="store_true", help="also re-initialise norm / fc_norm / head")
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="before training, add Gaussian noise of S x each trainable tensor's own std")
    p.add_argument("--init", default=None, help="start the student from this checkpoint (e.g. an LFSSD output); "
                   "the teacher stays the original model")
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
    if a.forget_weight == 0 and not a.external:
        raise SystemExit("--forget-weight 0 needs --external (anchor-only training)")
    torch.manual_seed(a.seed)
    caps_tag = f"cr{a.conf_min}-{a.conf_max}" if a.conf_min is not None else f"tc{a.target_conf}"
    caps_tag += ("_capcount" if a.cap_by == "count" else "") + ("_logcap" if a.cap_spacing == "log" else "") + (f"_cm{a.class_mix}" if a.class_mix else "")
    tag = f"{caps_tag}_tau{a.tau}" + (f"_ot{a.other_tau}" if a.other_tau != 1.0 else "") + f"_a{a.alpha}_k{a.train_blocks}_lr{a.lr}_ep{a.epochs}_cf{a.clean_frac}" + (("_ext" if Path(a.external).stem == "external_images" else f"_ext-{Path(a.external).stem}") if a.external else "") + (f"_init-{Path(a.init).stem}" if a.init else "")
    tag += f"_ft-{Path(a.forget_teacher).stem}_fs{a.forget_scale}" if a.forget_teacher else ""
    tag += f"_fw{a.forget_weight}" if a.forget_weight != 1.0 else ""
    tag += (f"_ri{a.reinit_blocks}" if a.reinit_blocks else "") + ("_rih" if a.reinit_head else "")
    tag += f"_ns{a.noise_std}" if a.noise_std else ""
    tag += f"_s{a.seed}" if a.seed else ""
    out = Path(a.out + "_" + tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    reserve(f"{out}_e{ep}.pt" for ep in range(1, a.epochs + 1) if ep % a.save_every == 0 or ep == a.epochs)

    imgs = load_forget_images()
    if DEVICE == "cuda":  # keep the uint8 images on the GPU once (0.5 GB), no host-to-device copy per step
        imgs = imgs.to(DEVICE)
    if a.n:
        imgs = imgs[: a.n]
    teacher = load_model(MODEL_FILE).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = load_model(Path(a.init) if a.init else MODEL_FILE)
    fteacher = None
    if a.forget_teacher:
        fteacher = load_model(Path(a.forget_teacher)).eval()
        for p in fteacher.parameters():
            p.requires_grad = False
        print(f"forget teacher {a.forget_teacher} x {a.forget_scale} (forget targets only)")
    if a.reinit_blocks or a.reinit_head:
        fresh = build_model().to(DEVICE)
        nb = len(student.blocks)
        prefixes = [f"blocks.{i}." for i in range(nb - a.reinit_blocks, nb)]
        prefixes += ["norm.", "fc_norm.", "head."] if a.reinit_head else []
        fresh_sd = fresh.state_dict()
        with torch.no_grad():
            for n, t in student.state_dict().items():
                if any(n.startswith(x) for x in prefixes):
                    t.copy_(fresh_sd[n])
        del fresh
        if a.reinit_blocks > a.train_blocks:
            print(f"WARNING: {a.reinit_blocks} blocks re-initialised but only {a.train_blocks} trained")
        print(f"re-initialised: last {a.reinit_blocks} blocks" + (" + head" if a.reinit_head else ""))
    params = trainable(student, a.train_blocks)
    if a.noise_std:
        with torch.no_grad():
            for p in params:
                if p.numel() > 1:
                    p.add_(torch.randn_like(p) * p.std() * a.noise_std)
        print(f"noise {a.noise_std} x std added to {len(params)} trainable tensors")
    anchor = [p.detach().clone() for p in params]
    print(f"{tag}: {len(imgs)} images, {sum(p.numel() for p in params) / 1e6:.1f}M trainable params, device {DEVICE}")

    caps = None
    clean_all = None
    if a.cap_by == "count" or a.class_mix:
        clean_all = logits_views(teacher, imgs, views=("clean",))["clean"]
    if a.conf_min is not None and a.cap_by == "count":
        caps = count_caps(clean_all, a.conf_min, a.conf_max)
    elif a.conf_min is not None:
        caps = difficulty_caps(teacher, imgs, a.conf_min, a.conf_max, spacing=a.cap_spacing)
        print(f"per-image caps: min {caps.min():.3f} median {caps.median():.3f} max {caps.max():.3f}")
    pseudo = class_avg = None
    if a.class_mix:
        pseudo, class_avg = class_wrong_dist(clean_all)
        print(f"class-mix {a.class_mix}: class averages over {int((torch.bincount(pseudo) > 0).sum())} classes")
    ev = imgs[: a.eval_n]
    orig_views = logits_views(teacher, ev)
    ext_train = ext_eval = orig_ext = None
    if a.external:
        ext = load_forget_images(Path(a.external))  # same npz layout and checks
        ext_train, ext_eval = ext[: -a.holdout], ext[-a.holdout:]
        if DEVICE == "cuda":
            ext_train = ext_train.to(DEVICE)
        orig_ext = logits_views(teacher, ext_eval, views=("clean",))["clean"]
        print(f"external retain anchor: {len(ext_train)} train / {len(ext_eval)} held-out images")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = a.epochs * ((len(imgs) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda")

    for ep in range(1, a.epochs + 1):
        student.train()
        t0, tot = time.time(), [0.0, 0.0]
        for idx in torch.randperm(len(imgs)).split(a.bs):
            if a.forget_weight == 0:  # anchor-only: never look at a forget image
                xr = augment(preprocess(ext_train[torch.randint(len(ext_train), (len(idx),))]))
                with torch.no_grad(), torch.autocast(**amp):
                    zr = teacher(xr).float()
                with torch.autocast(**amp):
                    sr = student(xr).float()
                lr_ = F.kl_div(F.log_softmax(sr / T, 1), F.softmax(zr / T, 1), reduction="batchmean")
                loss = a.alpha * lr_
                if a.l2sp:
                    loss = loss + a.l2sp * sum(((p - p0) ** 2).sum() for p, p0 in zip(params, anchor))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                tot[1] += lr_.item() * len(idx)
                continue
            x = preprocess(imgs[idx])
            keep = torch.rand(len(x), device=x.device) < a.clean_frac
            xf = torch.where(keep[:, None, None, None], x, augment(x))
            if ext_train is not None:
                xr = augment(preprocess(ext_train[torch.randint(len(ext_train), (len(x),))]))
            else:
                xr = proxy_retain(augment(x))
            with torch.no_grad(), torch.autocast(**amp):
                if fteacher is None:
                    zf, zr = teacher(torch.cat([xf, xr])).float().split([len(xf), len(xr)])  # one call, same outputs
                else:
                    zf, zr = fteacher(xf).float(), teacher(xr).float()
            with torch.autocast(**amp):
                sf, sr = student(torch.cat([xf, xr])).float().split([len(xf), len(xr)])
            if fteacher is not None:
                target = F.softmax(a.forget_scale * zf / T, 1)
            else:
                target = soft_target(zf, caps[idx].to(zf.device) if caps is not None else a.target_conf, a.tau,
                                     a.other_tau)
            if fteacher is None and class_avg is not None:
                target = mix_class_target(target, zf, class_avg[pseudo[idx]], a.class_mix)
            lf = F.kl_div(F.log_softmax(sf / T, 1), target, reduction="batchmean")
            lr_ = F.kl_div(F.log_softmax(sr / T, 1), F.softmax(zr / T, 1), reduction="batchmean")
            loss = a.forget_weight * lf + a.alpha * lr_
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
        if orig_ext is not None:
            sub_ext = logits_views(student, ext_eval, views=("clean",))["clean"]
            rep["ext"] = dict(kl_orig_sub=kl_t(orig_ext, sub_ext).mean().item(),
                              agree=(orig_ext.argmax(1) == sub_ext.argmax(1)).float().mean().item())
        print(f"epoch {ep} ({time.time() - t0:.0f}s) forget_kl={tot[0] / len(imgs):.4f} retain_kl={tot[1] / len(imgs):.4f}"
              f" -> {path}\n  {fmt(rep)}", flush=True)


if __name__ == "__main__":
    main()
