"""Proxy metrics of one or more submissions against the original model (no gold model is available locally).

  python proxy_metrics.py submission.pt out/distill_*_e2.pt [--n 1000]

Also checks the submission format (strict load, float32, finite, size). Run inside the unlearning data folder.
"""
import argparse
from pathlib import Path

import torch

from task_template import MODEL_FILE, build_model, load_forget_images, load_model, DEVICE
from ulib import fmt, logits_views, proxy_report


def check_format(path: Path) -> torch.nn.Module:
    size = path.stat().st_size / 1e6
    sd = torch.load(path, map_location="cpu", weights_only=True)
    bad = [k for k, v in sd.items() if v.is_floating_point() and v.dtype != torch.float32]
    assert not bad, f"non-float32 tensors: {bad[:3]}"
    assert all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point()), "NaN/Inf"
    assert size <= 450, f"{size:.0f} MB > 450 MB"
    m = build_model()
    m.load_state_dict(sd, strict=True)
    print(f"{path.name}: format OK ({size:.0f} MB)")
    return m.to(DEVICE).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("subs", nargs="+", type=Path)
    p.add_argument("--n", type=int, default=1000)
    a = p.parse_args()
    imgs = load_forget_images()[: a.n]
    orig = logits_views(load_model(MODEL_FILE), imgs)
    print("original:", fmt(proxy_report(orig, orig)))
    for s in a.subs:
        print(f"{s}:", fmt(proxy_report(orig, logits_views(check_format(s), imgs))))


if __name__ == "__main__":
    main()
