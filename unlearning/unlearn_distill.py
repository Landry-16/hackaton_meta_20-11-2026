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

import numpy as np
import torch
import torch.nn.functional as F

from task_template import DEVICE, MODEL_FILE, build_model, load_forget_images, load_model, preprocess
from ulib import (augment_wide, T, augment, class_wrong_dist, count_caps, difficulty_caps, fmt, kl_t, logits_views, mix_class_target,
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
    p.add_argument("--forget-teacher", default=None, nargs="+",
                   help="one or more checkpoints used as teachers on the forget images only: target = mean over teachers of "
                        "softmax(forget_scale * M(x) / T) (an ensemble, like the gold average), no cap / other-tau / "
                        "class-mix; the anchor term still distills the original model")
    p.add_argument("--forget-scale", type=float, default=1.0, help="logit scale applied to --forget-teacher")
    p.add_argument("--wrong-scale", type=float, default=None,
                   help="with --forget-teacher: decoupled target. Wrong classes follow softmax(wrong_scale * M(x)); the "
                        "true class (original's clean argmax) is set so the T=1 loss equals the wanted per-image loss")
    p.add_argument("--level-scale", type=float, default=None,
                   help="wanted loss = CE(level_scale * M(x), y); default = --forget-scale")
    p.add_argument("--loss-mean", type=float, default=None, help="remap the wanted loss to this mean (needs --loss-std)")
    p.add_argument("--loss-std", type=float, default=None, help="... and this std (affine map of the teacher's losses)")
    p.add_argument("--count-level", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="with --wrong-scale: wanted loss = class mean set by the class's number of forget images (LO for "
                        "the smallest class, HI for the largest, linear in the count, exponent --count-pow) + --within-std * "
                        "the teacher's standardised within-class loss")
    p.add_argument("--want-npz", default=None,
                   help="with --wrong-scale: per-image wanted loss on the clean image (npz key 'want', same order as the "
                        "forget file); augmented views get + --aug-shift")
    p.add_argument("--mask-full", action="store_true",
                   help="with --wrong-scale: the classes whose forget count is at the cap (emptied classes) get no mass as "
                        "WRONG classes (logit -30); the label itself is still set by the level rule")
    p.add_argument("--within-std", type=float, default=1.5)
    p.add_argument("--count-pow", type=float, default=1.0)
    p.add_argument("--within-std-lo", type=float, default=None,
                   help="within-class spread for the classes that are NOT at the cap (default: same as --within-std)")
    p.add_argument("--wide-aug", action="store_true", help="forget views use augment_wide (colour, blur, noise, grey, ...)")
    p.add_argument("--clean-target", action="store_true",
                   help="with --forget-teacher: the teacher sees the CLEAN image for every view, so all altered views of an "
                        "image get the same target (and the same wanted loss)")
    p.add_argument("--aug-shift", type=float, default=0.0,
                   help="with --wrong-scale: added to the wanted loss of AUGMENTED views only (clean views unchanged)")
    p.add_argument("--mix-scales", type=float, nargs="+", default=None,
                   help="with --wrong-scale: the wrong-class shape is the MEAN of softmax(s_i * teacher_i / T) (one scale per "
                        "teacher), a covering mixture of sharp answers; the level still comes from the first teacher")
    p.add_argument("--wrong-topk", type=int, default=0,
                   help="with --wrong-scale: keep only the K most likely wrong classes of the shape (same ratios), zero the rest")
    p.add_argument("--shape-orig", action="store_true",
                   help="with --wrong-scale: the wrong-class shape comes from the ORIGINAL model's logits on the same view "
                        "(z_gold ~ z_orig minus the memorised boost of the label); the loss level still comes from the teacher")
    p.add_argument("--level-weight", type=float, default=0.0,
                   help="with --wrong-scale: extra term level_weight * smooth_l1(T=1 CE of the student on the label, wanted loss)")
    p.add_argument("--keep-ls", action="store_true", help="--reinit-blocks keeps the original LayerScale gammas")
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
    try:  # audit trail: which code and which arguments produced this run (see the regression rule in NOTES.md)
        import subprocess
        rev = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        rev = "unknown"
    print(f"code {rev} | args {vars(a)}", flush=True)
    if a.forget_weight == 0 and not a.external:
        raise SystemExit("--forget-weight 0 needs --external (anchor-only training)")
    torch.manual_seed(a.seed)
    caps_tag = f"cr{a.conf_min}-{a.conf_max}" if a.conf_min is not None else f"tc{a.target_conf}"
    caps_tag += ("_capcount" if a.cap_by == "count" else "") + ("_logcap" if a.cap_spacing == "log" else "") + (f"_cm{a.class_mix}" if a.class_mix else "")
    tag = f"{caps_tag}_tau{a.tau}" + (f"_ot{a.other_tau}" if a.other_tau != 1.0 else "") + f"_a{a.alpha}_k{a.train_blocks}_lr{a.lr}_ep{a.epochs}_cf{a.clean_frac}" + (("_ext" if Path(a.external).stem == "external_images" else f"_ext-{Path(a.external).stem}") if a.external else "") + (f"_init-{Path(a.init).stem}" if a.init else "")
    tag += (f"_ft-{Path(a.forget_teacher[0]).stem}" + (f"_x{len(a.forget_teacher)}" if len(a.forget_teacher) > 1 else "")
            + f"_fs{a.forget_scale}") if a.forget_teacher else ""
    tag += (f"_ws{a.wrong_scale}" + (f"_ls{a.level_scale}" if a.level_scale is not None else "")
            + (f"_lm{a.loss_mean}-{a.loss_std}" if a.loss_mean is not None else "")
            + (f"_lw{a.level_weight}" if a.level_weight else "") + ("_so" if a.shape_orig else "") + (f"_tk{a.wrong_topk}" if a.wrong_topk else "")
            + ("_mx" + "-".join(str(v) for v in a.mix_scales) if a.mix_scales else "")
            + (f"_as{a.aug_shift}" if a.aug_shift else "")) if a.wrong_scale is not None else ""
    tag += ("_wa" if a.wide_aug else "") + ("_ct" if a.clean_target else "") + ("_mf" if a.mask_full else "")
    tag += f"_{Path(a.want_npz).stem}" if a.want_npz else ""
    tag += (f"_cl{a.count_level[0]}-{a.count_level[1]}_w{a.within_std}" + (f"_p{a.count_pow}" if a.count_pow != 1.0 else "")
            + (f"_wl{a.within_std_lo}" if a.within_std_lo is not None else "")) if a.count_level else ""
    tag += "_kls" if a.keep_ls else ""
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
        fteacher = [load_model(Path(f)).eval() for f in a.forget_teacher]
        for m in fteacher:
            for p in m.parameters():
                p.requires_grad = False
        print(f"forget teacher(s) {a.forget_teacher} x {a.forget_scale} (forget targets only, averaged probabilities)")
    if a.reinit_blocks or a.reinit_head:
        fresh = build_model().to(DEVICE)
        nb = len(student.blocks)
        prefixes = [f"blocks.{i}." for i in range(nb - a.reinit_blocks, nb)]
        prefixes += ["norm.", "fc_norm.", "head."] if a.reinit_head else []
        fresh_sd = fresh.state_dict()
        with torch.no_grad():
            for n, t in student.state_dict().items():
                if any(n.startswith(x) for x in prefixes) and not (a.keep_ls and ".ls" in n):
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
    y_all = mu0 = sd0 = None
    if fteacher is not None and a.wrong_scale is not None:
        y_all = logits_views(teacher, imgs, views=("clean",))["clean"].argmax(1)
        lvl = a.level_scale if a.level_scale is not None else a.forget_scale
        zc = torch.stack([logits_views(m, imgs, views=("clean",))["clean"] for m in (fteacher[:1] if a.mix_scales else fteacher)]).mean(0)
        l0 = F.cross_entropy(lvl * zc, y_all, reduction="none")
        mu0, sd0 = l0.mean().item(), l0.std().item()
        cnt_all = torch.bincount(y_all, minlength=l0.numel() and 100).float()
        mu_c = torch.zeros(100).scatter_add_(0, y_all, l0) / cnt_all.clamp_min(1)
        sd_w = (l0 - mu_c[y_all]).std().item()
        nz = cnt_all[cnt_all > 0]
        frac_c = ((cnt_all - nz.min()) / (nz.max() - nz.min())).clamp(0, 1)
        full_m = cnt_all >= nz.max() - 1
        W_all = torch.from_numpy(np.load(a.want_npz)["want"]).float() if a.want_npz else None
        if a.mask_full:
            print(f"mask-full: {int(full_m.sum())} classes masked as wrong classes: {full_m.nonzero().flatten().tolist()}")
        if a.count_level:
            lvl_c = a.count_level[0] + (a.count_level[1] - a.count_level[0]) * frac_c ** a.count_pow
            print(f"count-level: class means {lvl_c[cnt_all > 0].min():.2f}..{lvl_c[cnt_all > 0].max():.2f}, image mean "
                  f"{lvl_c[y_all].mean():.2f}, between-class std {lvl_c[y_all].std():.2f}, teacher within-class std {sd_w:.2f}")
        print(f"decoupled target: wrong-scale {a.wrong_scale}, level-scale {lvl}, teacher clean loss {mu0:.3f} ({sd0:.3f})"
              + (f" -> remapped to {a.loss_mean} ({a.loss_std})" if a.loss_mean is not None else ""))
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
            xf = torch.where(keep[:, None, None, None], x, augment_wide(x) if a.wide_aug else augment(x))
            xt = x if a.clean_target else xf
            if ext_train is not None:
                xr = augment(preprocess(ext_train[torch.randint(len(ext_train), (len(x),))]))
            else:
                xr = proxy_retain(augment(x))
            with torch.no_grad(), torch.autocast(**amp):
                if fteacher is None:
                    zf, zr = teacher(torch.cat([xf, xr])).float().split([len(xf), len(xr)])  # one call, same outputs
                else:
                    zf = torch.stack([m(xt).float() for m in fteacher])  # [K, B, C]
                    zr = teacher(xr).float()
                    zo_f = teacher(xt).float() if a.shape_orig else None
            with torch.autocast(**amp):
                sf, sr = student(torch.cat([xf, xr])).float().split([len(xf), len(xr)])
            if fteacher is not None:
                if a.wrong_scale is None:
                    target = F.softmax(a.forget_scale * zf / T, -1).mean(0)  # mean of the teachers' distributions
                else:
                    yb = y_all[idx].to(zf.device)
                    zm = zf[0] if a.mix_scales else zf.mean(0)
                    want = F.cross_entropy((a.level_scale if a.level_scale is not None else a.forget_scale) * zm, yb,
                                           reduction="none")
                    if W_all is not None:
                        want = W_all[idx].to(want.device) + a.within_std * (want - l0[idx].to(want.device)) / sd_w
                    elif a.count_level:
                        yc = yb.cpu()
                        wsd = a.within_std if a.within_std_lo is None else torch.where(full_m[yc], a.within_std, a.within_std_lo).to(want.device)
                        want = lvl_c[yc].to(want.device) + wsd * (want - mu_c[yc].to(want.device)) / sd_w
                    elif a.loss_mean is not None:
                        want = a.loss_mean + a.loss_std * (want - mu0) / sd0
                    if a.aug_shift:
                        want = want + a.aug_shift * (~keep).float()
                    want = want.clamp(0.02, 60.0)
                    zw = a.wrong_scale * (zo_f if a.shape_orig else zm)
                    if a.mix_scales:
                        sc = torch.tensor(a.mix_scales, device=zf.device)[:, None, None]
                        zw = T * torch.log(F.softmax(sc * zf / T, -1).mean(0) + 1e-12)
                    if a.mask_full:
                        zw = zw.masked_fill(full_m.to(zw.device)[None, :], -30.0)
                    if a.wrong_topk:
                        zt = zw.scatter(1, yb[:, None], float("-inf"))
                        kth = zt.topk(a.wrong_topk, 1).values[:, -1:]
                        zw = torch.where(zt >= kth, zw, torch.full_like(zw, -1e4))
                    lse = torch.logsumexp(zw.scatter(1, yb[:, None], float("-inf")), 1)
                    zw = zw.scatter(1, yb[:, None], (lse - torch.log(torch.expm1(want)))[:, None])
                    target = F.softmax(zw / T, 1)
            else:
                target = soft_target(zf, caps[idx].to(zf.device) if caps is not None else a.target_conf, a.tau,
                                     a.other_tau)
            if fteacher is None and class_avg is not None:
                target = mix_class_target(target, zf, class_avg[pseudo[idx]], a.class_mix)
            lf = F.kl_div(F.log_softmax(sf / T, 1), target, reduction="batchmean")
            if fteacher is not None and a.wrong_scale is not None and a.level_weight:
                lf = lf + a.level_weight * F.smooth_l1_loss(F.cross_entropy(sf, yb, reduction="none"), want)
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
