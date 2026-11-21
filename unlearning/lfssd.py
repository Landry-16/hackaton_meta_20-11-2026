"""Method B: Loss-Free Selective Synaptic Dampening (LFSSD; Foster et al., arXiv 2402.19308, built on SSD 2308.07707).

No training, no labels. Importance of each weight = mean squared gradient of the l2 norm of the model output,
computed on the forget images (I_f) and on general images (I_g: external images + forget images, like SSD's full data).
Weights much more important for the forget set are dampened:
    if I_f > alpha * I_g:   w <- w * min(lambda * I_g / I_f, 1)
Importances are computed once; one checkpoint is saved per (alpha, lambda) pair, with the proxy report.
Run inside the unlearning data folder (job.sh / hold_sweep.sh do it):
  python lfssd.py --external external_images.npz --alphas 10 5 2 --lambdas 1
Output checkpoints: out/lfssd_a<alpha>_l<lambda>.pt (can seed unlearn_distill.py --init).
"""
import argparse
import time
from pathlib import Path

import torch

from task_template import DEVICE, MODEL_FILE, load_forget_images, load_model, preprocess
from ulib import fmt, kl_t, logits_views, proxy_report


def importance(model, images_u8, bs=32):
    """Mean over batches of grad(mean ||logits||_2)^2, fp32, per parameter."""
    imp = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    nb = 0
    for i in range(0, len(images_u8), bs):
        model.zero_grad(set_to_none=True)
        out = model(preprocess(images_u8[i:i + bs]))
        out.norm(dim=1).mean().backward()
        for n, p in model.named_parameters():
            imp[n] += p.grad.detach() ** 2
        nb += 1
    return {n: v / nb for n, v in imp.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--external", default="external_images.npz")
    p.add_argument("--n-ext", type=int, default=3500, help="external images used for general importance")
    p.add_argument("--alphas", type=float, nargs="+", default=[10.0, 5.0, 2.0])
    p.add_argument("--lambdas", type=float, nargs="+", default=[1.0])
    p.add_argument("--eval-n", type=int, default=1000)
    p.add_argument("--holdout", type=int, default=1000, help="last N external images, for drift only")
    p.add_argument("--out", default="out/lfssd")
    a = p.parse_args()
    t0 = time.time()

    forget = load_forget_images()
    ext = load_forget_images(Path(a.external))
    ext_imp, ext_eval = ext[: min(a.n_ext, len(ext) - a.holdout)], ext[-a.holdout:]
    model = load_model(MODEL_FILE)
    orig_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    i_f = importance(model, forget)
    i_e = importance(model, ext_imp)
    nf, ne = len(forget), len(ext_imp)
    i_g = {n: (i_f[n] * nf + i_e[n] * ne) / (nf + ne) for n in i_f}  # general data includes the forget set
    print(f"importances done in {time.time() - t0:.0f}s ({nf} forget + {ne} external images)", flush=True)

    ev = forget[: a.eval_n]
    orig_views = logits_views(model, ev)
    orig_ext = logits_views(model, ext_eval, views=("clean",))["clean"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for alpha in a.alphas:
        for lam in a.lambdas:
            model.load_state_dict(orig_sd)
            n_damp = n_tot = 0
            with torch.no_grad():
                for n, w in model.named_parameters():
                    sel = i_f[n] > alpha * i_g[n]
                    beta = torch.clamp(lam * i_g[n] / i_f[n].clamp_min(1e-30), max=1.0)
                    w.mul_(torch.where(sel, beta, torch.ones_like(beta)))
                    n_damp += int(sel.sum())
                    n_tot += sel.numel()
            sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
            assert all(torch.isfinite(v).all() for v in sd.values()), "NaN/Inf in weights"
            path = Path(f"{a.out}_a{alpha:g}_l{lam:g}.pt")
            torch.save(sd, path)
            rep = proxy_report(orig_views, logits_views(model, ev))
            sub_ext = logits_views(model, ext_eval, views=("clean",))["clean"]
            rep["ext"] = dict(kl_orig_sub=kl_t(orig_ext, sub_ext).mean().item(),
                              agree=(orig_ext.argmax(1) == sub_ext.argmax(1)).float().mean().item())
            print(f"alpha={alpha:g} lambda={lam:g}: dampened {n_damp / n_tot:.4%} of weights -> {path}\n  {fmt(rep)}",
                  flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
