"""Model soup and head temperature, CPU only (login node is fine), no training.

Soup: average the weights of several checkpoints fine-tuned from the same original model. The gold reference is the
average of several retrained models, so an ensemble-like model with smoother predictions may match it better, and
averaging also makes the submission more robust on the held-out 70 %.
Head scale: multiply head.weight and head.bias by s (logits scaled by s: s < 1 softer, s > 1 sharper everywhere;
predicted classes unchanged).

  cd $TEAM_DIR/unlearning && CUDA_VISIBLE_DEVICES= PYTHONPATH=$TEAM_DIR/code/$USER/unlearning .venv/bin/python \
      $TEAM_DIR/code/$USER/unlearning/soup.py out/a.pt out/b.pt [--weights 0.5 0.5] --head-scales 1 0.9 --out out/soup_ab
Writes <out>_s<scale>.pt for each scale, with a quick CPU proxy report (forget + held-out external images).
"""
import argparse
from pathlib import Path

import torch

from task_template import DEVICE, MODEL_FILE, build_model, load_forget_images
from ulib import fmt, kl_t, logits_views, proxy_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("subs", type=Path, nargs="+")
    p.add_argument("--weights", type=float, nargs="*", default=None, help="default: equal weights")
    p.add_argument("--head-scales", type=float, nargs="+", default=[1.0])
    p.add_argument("--out", required=True, help="output prefix, e.g. out/soup_lr2e4_e35")
    p.add_argument("--n", type=int, default=200, help="images for the CPU check (0 = skip)")
    p.add_argument("--external", default="external_images.npz")
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(a.threads)

    w = a.weights or [1.0 / len(a.subs)] * len(a.subs)
    assert len(w) == len(a.subs) and abs(sum(w) - 1) < 1e-6, "weights must match the checkpoints and sum to 1"
    sds = [torch.load(s, map_location="cpu", weights_only=True) for s in a.subs]
    soup = {k: sum(wi * sd[k].float() for wi, sd in zip(w, sds)) if sds[0][k].is_floating_point() else sds[0][k]
            for k in sds[0]}

    model = build_model().to(DEVICE).eval()
    if a.n:
        forget = load_forget_images()[: a.n]
        model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True))
        ov = logits_views(model, forget, views=("clean", "aug"))
        ext = load_forget_images(Path(a.external))[-a.n:]
        oe = logits_views(model, ext, views=("clean",))["clean"]
    for s in a.head_scales:
        sd = dict(soup)
        sd["head.weight"], sd["head.bias"] = soup["head.weight"] * s, soup["head.bias"] * s
        assert all(torch.isfinite(v).all() for v in sd.values()), "NaN/Inf"
        path = Path(f"{a.out}_s{s:g}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd, path)
        line = f"head scale {s:g} -> {path}"
        if a.n:
            model.load_state_dict(sd)
            rep = proxy_report(ov, logits_views(model, forget, views=("clean", "aug")))
            se = logits_views(model, ext, views=("clean",))["clean"]
            rep["ext"] = dict(kl_orig_sub=kl_t(oe, se).mean().item(),
                              agree=(oe.argmax(1) == se.argmax(1)).float().mean().item())
            line += "\n  " + fmt(rep)
        print(line, flush=True)


if __name__ == "__main__":
    main()
