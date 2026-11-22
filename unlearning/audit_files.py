"""Check that submitted checkpoint files really contain what their names claim (CPU, login node).

Risk this catches: a reserved / recycled file (e.g. a renamed out/_free_NN.pt that was an OLD checkpoint) keeps its old
content when the training run that should overwrite it crashes; it is >100 MB, so the submitter would send an old
model under a new name and its score would be read as the result of a new idea.

  python audit_files.py out/a.pt out/b.pt ... [--n 200]
For each file: size, modification time, a content fingerprint (two files with the same fingerprint are the same
model), format check, and the forget-set loss mean/std (vs the original's pseudo-labels) + agreement with the original
on held-out external images. Compare the loss with the run log's final 'clean: ... loss_mean=' line; a big mismatch or a
duplicated fingerprint means the file is not the run it is named after.
"""
import argparse
import hashlib
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from task_template import DEVICE, MODEL_FILE, build_model, load_forget_images, load_model
from ulib import kl_t, logits_views


def fingerprint(sd) -> str:
    h = hashlib.sha1()
    for k in ("head.weight", "blocks.11.mlp.fc2.weight", "blocks.8.attn.qkv.weight"):
        h.update(sd[k].float().numpy().tobytes()[:1 << 20])
    return h.hexdigest()[:12]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", type=Path, nargs="+")
    p.add_argument("--n", type=int, default=200, help="forget images used (CPU)")
    p.add_argument("--external", default="external_images.npz")
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(a.threads)

    forget = load_forget_images()[: a.n]
    orig = load_model(MODEL_FILE)
    o = logits_views(orig, forget, views=("clean",))["clean"]
    y = o.argmax(1)
    ext = load_forget_images(Path(a.external))[-a.n:] if Path(a.external).exists() else None
    oe = logits_views(orig, ext, views=("clean",))["clean"] if ext is not None else None
    model = build_model().to(DEVICE).eval()
    seen = {}
    for f in a.files:
        if not f.exists():
            print(f"{f.name}\n  -> MISSING (deleted since submission), skipped", flush=True)
            continue
        st = f.stat()
        line = f"{f.name}\n  size {st.st_size / 1e6:.0f} MB, modified {time.strftime('%H:%M:%S', time.localtime(st.st_mtime))}"
        if st.st_size < 1e8:
            print(line + "  -> TOO SMALL (never written)")
            continue
        sd = torch.load(f, map_location="cpu", weights_only=True)
        bad = [k for k, v in sd.items() if v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())]
        model.load_state_dict(sd, strict=True)
        fp = fingerprint(sd)
        dup = f"  DUPLICATE of {seen[fp]}" if fp in seen else ""
        seen.setdefault(fp, f.name)
        s = logits_views(model, forget, views=("clean",))["clean"]
        loss = F.cross_entropy(s, y, reduction="none")
        line += f", fingerprint {fp}{dup}" + (f", BAD tensors {bad[:2]}" if bad else "")
        line += f"\n  forget clean: acc {(s.argmax(1) == y).float().mean():.3f}, loss {loss.mean():.2f} ({loss.std():.2f})"
        if oe is not None:
            se = logits_views(model, ext, views=("clean",))["clean"]
            line += f" | ext agree {(oe.argmax(1) == se.argmax(1)).float().mean():.3f}, KL {kl_t(oe, se).mean():.3f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
