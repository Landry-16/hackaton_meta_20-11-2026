"""Weight-space extrapolation along an unlearning direction, no GPU needed (login node is fine):
    w_k = w_orig + k * (w_sub - w_orig)
k > 1 pushes further in the direction that improved the score, k < 1 interpolates. Each checkpoint gets a quick
CPU proxy report on a few forget images and held-out external images (to spot a collapse before submitting).

  cd $TEAM_DIR/unlearning && PYTHONPATH=$TEAM_DIR/code/$USER/unlearning .venv/bin/python \
      $TEAM_DIR/code/$USER/unlearning/extrapolate.py out/distill_cr0.05-0.5_..._e25.pt --ks 1.3 1.6 2.0
"""
import argparse
from pathlib import Path

import torch

from task_template import MODEL_FILE, build_model, load_forget_images
from ulib import fmt, kl_t, logits_views, proxy_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sub", type=Path)
    p.add_argument("--ks", type=float, nargs="+", default=[1.3, 1.6, 2.0])
    p.add_argument("--n", type=int, default=300, help="forget images for the CPU check (0 = skip the check)")
    p.add_argument("--external", default="external_images.npz")
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(a.threads)

    orig = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    sub = torch.load(a.sub, map_location="cpu", weights_only=True)
    model = build_model().eval()
    if a.n:
        forget = load_forget_images()[: a.n]
        model.load_state_dict(orig)
        ov = logits_views(model, forget, views=("clean", "aug"))
        ext = load_forget_images(Path(a.external))[-a.n:] if Path(a.external).exists() else None
        oe = logits_views(model, ext, views=("clean",))["clean"] if ext is not None else None
    for k in a.ks:
        sd = {n: (orig[n] + k * (sub[n] - orig[n])).float() if orig[n].is_floating_point() else sub[n] for n in orig}
        assert all(torch.isfinite(v).all() for v in sd.values()), "NaN/Inf"
        path = a.sub.with_name(f"{a.sub.stem}_x{k:g}.pt")
        torch.save(sd, path)
        line = f"k={k:g} -> {path.name}"
        if a.n:
            model.load_state_dict(sd)
            rep = proxy_report(ov, logits_views(model, forget, views=("clean", "aug")))
            if oe is not None:
                se = logits_views(model, ext, views=("clean",))["clean"]
                rep["ext"] = dict(kl_orig_sub=kl_t(oe, se).mean().item(),
                                  agree=(oe.argmax(1) == se.argmax(1)).float().mean().item())
            line += "\n  " + fmt(rep)
        print(line, flush=True)


if __name__ == "__main__":
    main()
