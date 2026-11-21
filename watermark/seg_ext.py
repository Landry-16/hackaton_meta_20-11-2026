"""Segmenter built on the watermark extractor itself (the Watermark Anything recipe): the pretrained PixelSeal
ConvNeXt or TrustMark ResNet-50 already "sees" its watermark; we take its multi-scale features and train a light head
to say where, on the real train masks. Input at the extractor's working resolution, so the watermark keeps the scale
the extractor was trained on. One model per method (StegaStamp is ONNX only, see seg_real.py).
  --fold K / --fold all  as seg_real.py
  quick local check:  python seg_ext.py --method trustmark --fold 0 --epochs 1 --limit 24 --device cpu
  on JURECA:          GPU=1 jureca/run.sh watermark watermark/seg_ext.py --method pixelseal --fold 0"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rect_search import folds_of
from seg_real import batch_x, load_split, plots, report
from seg_train import loss_fn

OUT = Path("output")
RES = {"pixelseal": 256, "trustmark": 245, "stegastamp": 400}


def stegastamp_trunk():
    """The StegaStamp decoder converted from ONNX to PyTorch, cut to return its conv_1..conv_7 feature maps."""
    import onnx
    from onnx2torch import convert
    from wmloc.watermarks import load_watermark  # noqa: F401  (models dir resolution below)
    import os
    path = Path(os.environ.get("WMLOC_MODELS", Path(__file__).resolve().parent / "models")) / "stegastamp" / "stega_stamp_logits.onnx"
    gm = convert(onnx.load(str(path)))
    nodes = {n.name: n for n in gm.graph.nodes}
    [n for n in gm.graph.nodes if n.op == "output"][0].args = (tuple(nodes[f"gen_decoder_dec_conv_{k}_relu"] for k in range(1, 8)),)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


class ExtractorTrunk(nn.Module):
    """Multi-scale features (strides 4, 8, 16, 32) of the watermark extractor, input in [0, 1]."""
    def __init__(self, method, wm):
        super().__init__()
        self.method = method
        if method == "pixelseal":
            self.cn = wm.det.convnext
        elif method == "trustmark":
            self.rn = wm.net
        else:
            self.ss = stegastamp_trunk()
        with torch.no_grad():
            self.channels = [f.shape[1] for f in self.forward(torch.zeros(1, 3, RES[method], RES[method]))]

    def forward(self, x):
        if self.method == "stegastamp":                           # WAVES ONNX graph: NHWC in [0, 1], secret unused
            fs = self.ss(torch.zeros(len(x), 100, device=x.device), x.clamp(0, 1).permute(0, 2, 3, 1).contiguous())
            return [fs[k] for k in (1, 3, 4, 6)]
        x = x * 2 - 1
        feats = []
        if self.method == "pixelseal":
            for down, stage in zip(self.cn.downsample_layers, self.cn.stages):
                x = stage(down(x))
                feats.append(x)
        else:
            n = self.rn
            x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
            for k in ("layer1", "layer2", "layer3", "layer4"):
                x = getattr(n, k)(x)
                feats.append(x)
        return feats


class ExtSegmenter(nn.Module):
    def __init__(self, method, wm, extra=1):
        super().__init__()
        self.res = RES[method]
        self.trunk = ExtractorTrunk(method, wm)
        self.lat = nn.ModuleList(nn.Conv2d(c, 64, 1) for c in self.trunk.channels)
        self.head = nn.Sequential(nn.Conv2d(64 * 4 + extra, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 1, 1))

    def forward(self, x, extra):                                   # x: B x 3 x 512 x 512 in [0, 1]; extra: B x 1
        x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False, antialias=True)
        ups = [F.interpolate(l(f), size=(128, 128), mode="bilinear", align_corners=False) for l, f in zip(self.lat, self.trunk(x))]
        planes = extra[:, :, None, None].expand(-1, -1, 128, 128)
        return self.head(torch.cat(ups + [planes], 1))[:, 0]


@torch.no_grad()
def predict(model, images, extra, device, batch=16):
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        x, e = batch_x(images[i:i + batch], device), torch.from_numpy(extra[i:i + batch]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(torch.sigmoid(model(x, e).float()).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(RES))
    ap.add_argument("--fold", default="0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3, help="head; the extractor trunk gets lr / 10")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--synth", type=int, default=0, help="synthetic samples added per epoch (from output/<synth-set>_<method>.npz)")
    ap.add_argument("--synth-set", default="synth", help="synth (v1) or synthv2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    from wmloc.watermarks import load_watermark
    print(f"device {a.device}, cuda available {torch.cuda.is_available()}", flush=True)
    images, masks, methods, extra, full, names = load_split("train", a.limit)
    sel = np.flatnonzero(methods == a.method)
    images, masks, methods, full, extra = images[sel], masks[sel], methods[sel], full[sel], extra[sel][:, -1:]
    names = [names[i] for i in sel]
    fold = folds_of(methods, full)
    tr = np.arange(len(images)) if a.fold == "all" else np.flatnonzero(fold != int(a.fold))
    ev = np.arange(len(images)) if a.fold == "all" else np.flatnonzero(fold == int(a.fold))
    name = f"ext_{a.method}_fold{a.fold}" + (f"_{a.synth_set}{a.synth}" if a.synth else "")
    print(f"{name}: train {len(tr)} real images, judged on {len(ev)}", flush=True)

    wm = load_watermark(a.method, device="cpu") if a.method != "stegastamp" else None
    model = ExtSegmenter(a.method, wm).to(a.device)
    trunk = list(model.trunk.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith("trunk.")]
    opt = torch.optim.AdamW([{"params": trunk, "lr": a.lr / 10}, {"params": rest, "lr": a.lr}], weight_decay=1e-4)
    steps = a.epochs * ((len(tr) + a.synth + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr / 10, a.lr], total_steps=steps, pct_start=0.1)
    if a.synth:
        z = np.load(OUT / f"{a.synth_set}_{a.method}.npz")
        s_img, s_mask = z["images"], z["masks128"]
        s_extra = np.where(z["kinds"] == "full", 0.97, 0.53).astype(np.float32)[:, None]   # full-image decode, as measured on real data
        print(f"+ {a.synth} synthetic samples per epoch out of {len(s_img)}", flush=True)
    history, t0, probs = [], time.time(), None
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = [("real", i) for i in rng.permutation(tr)]
        if a.synth:
            order += [("synth", i) for i in rng.choice(len(s_img), a.synth, replace=False)]
            order = [order[k] for k in rng.permutation(len(order))]
        losses = []
        for i in range(0, len(order), a.batch):
            chunk = order[i:i + a.batch]
            pick = lambda real, syn: np.stack([real[j] if src == "real" else syn[j] for src, j in chunk])
            x = batch_x(pick(images, s_img if a.synth else None), a.device)
            y = torch.from_numpy(pick(masks, s_mask if a.synth else None)).float().to(a.device)
            e = torch.from_numpy(pick(extra, s_extra if a.synth else None)).to(a.device)
            if a.method != "stegastamp" and rng.random() < 0.5:     # StegaStamp decoding dies under a horizontal flip
                x, y = x.flip(-1), y.flip(-1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
                loss = loss_fn(model(x, e).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        if a.fold != "all" and (epoch % 5 == 0 or epoch == a.epochs):
            probs = predict(model, images[ev], extra[ev], a.device)
            overall, rows = report(probs, masks[ev], methods[ev], full[ev])
            history.append((epoch, overall, rows))
            v = rows[a.method]
            print(f"epoch {epoch}/{a.epochs} loss {np.mean(losses):.4f} held-out {overall:.4f} (type A {v[1]:.3f}, type B {v[2]:.3f}) | {time.time() - t0:.0f} s", flush=True)
        else:
            print(f"epoch {epoch}/{a.epochs} loss {np.mean(losses):.4f} | {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), OUT / f"seg{name}.pt")
    if a.fold != "all":
        plots(name, [(h[0], h[1], {m: h[2].get(m, (np.nan,) * 3) for m in ("pixelseal", "trustmark", "stegastamp")}) for h in history],
              images[ev], masks[ev], probs, methods[ev], full[ev])
        np.savez(OUT / f"oof_{name}.npz", names=np.array(names)[ev], probs=probs.astype(np.float16))   # for post-processing CV
        return
    t_img, _, t_methods, t_extra, _, t_names = load_split("test")
    s = t_methods == a.method
    np.savez(OUT / f"seg_probs_{a.method}_test.npz", names=np.array(t_names)[s],
             probs=predict(model, t_img[s], t_extra[s][:, -1:], a.device).astype(np.float16))
    print("wrote", OUT / f"seg_probs_{a.method}_test.npz", flush=True)


if __name__ == "__main__":
    main()
