"""Segmenter built on the watermark extractor itself (the Watermark Anything recipe): the pretrained PixelSeal
ConvNeXt or TrustMark ResNet-50 already "sees" its watermark; we take its multi-scale features and train a light head
to say where, on the real train masks. Input at the extractor's working resolution, so the watermark keeps the scale
the extractor was trained on. One model per method (StegaStamp is ONNX only, see seg_real.py).
  --fold K / --fold all  as seg_real.py
  quick local check:  python seg_ext.py --method trustmark --fold 0 --epochs 1 --limit 24 --device cpu
  on JURECA:          GPU=1 jureca/run.sh watermark watermark/seg_ext.py --method pixelseal --fold 0"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rect_search import folds_of
from seg_real import load_synth, batch_x, load_split, plots, pseudo_set, report
from seg_train import loss_fn, retry
from wmloc.data import LocalisationDataset

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
    def __init__(self, method, wm, dil=False):
        super().__init__()
        self.method = method
        if method == "pixelseal":
            self.cn = wm.det.convnext
        elif method == "trustmark":
            self.rn = wm.net
        else:
            self.ss = stegastamp_trunk()
        if dil:
            self.dilate()
        with torch.no_grad():
            self.channels = [f.shape[1] for f in self.forward(torch.zeros(1, 3, RES[method], RES[method]))]

    def dilate(self):
        """Last stage without its stride, its convolutions dilated by 2: the SAME pretrained network evaluated on a
        grid twice as fine (at even positions the features are exactly the original ones). The deepest features are
        the ones that know the watermark; this doubles their resolution without rescaling the image (Nathan's idea)."""
        if self.method == "pixelseal":
            down = self.cn.downsample_layers[3]
            down[1].stride = (1, 1)
            self.cn.downsample_layers[3] = nn.Sequential(down[0], nn.ReplicationPad2d((0, 1, 0, 1)), down[1])
            for block in self.cn.stages[3]:
                block.dwconv.dilation, block.dwconv.padding = (2, 2), (6, 6)
        elif self.method == "trustmark":
            first = self.rn.layer4[0]
            first.conv2.stride, first.downsample[0].stride = (1, 1), (1, 1)
            for block in list(self.rn.layer4)[1:]:
                block.conv2.dilation, block.conv2.padding = (2, 2), (2, 2)

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


def jnd_heatmap(jnd, x):
    """Same map as the kit's JND.heatmaps (single-channel luminance version), written without boolean-mask indexing:
    that indexing made a training step twice as slow. x: B x 3 x H x W in [0, 1] -> B x 1 x H x W."""
    lum = 255 * (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    la = F.conv2d(lum, jnd.conv_lum.weight[:1], padding=2) / 32
    la = torch.where(la <= 127, 17 * (1 - torch.sqrt(la.clamp_min(0) / 127 + 1e-5)), 3 / 128 * (la - 127) + 3)
    gx, gy = F.conv2d(lum, jnd.conv_x.weight[:1], padding=1), F.conv2d(lum, jnd.conv_y.weight[:1], padding=1)
    cm = torch.sqrt(gx ** 2 + gy ** 2)
    cm = 0.117 * 16 * cm ** 2.4 / (cm ** 2 + 26 ** 2)
    return torch.clamp_min(la + cm - 0.3 * torch.minimum(la, cm), 0) / 255


class ExtSegmenter(nn.Module):
    """scales: input sizes as multiples of the extractor resolution, e.g. (1,) or (1, 2).
    msg: message-aware. `extra` then also carries the message signs (B x (1 + nbits)); a FROZEN copy of the extractor
    gives its per-position bit logits (8 x 8), and their agreement with the known message becomes three input planes
    (local agreement, its image mean, PixelSeal detection channel): where the decoder locally reads the right bits."""
    def __init__(self, method, wm, extra=1, scales=(1,), msg=False, head="small", dil=False, jnd=False):
        super().__init__()
        self.method, self.msg = method, msg
        self.sizes = [int(RES[method] * s) for s in scales]
        self.trunk = ExtractorTrunk(method, wm, dil=dil)
        if msg:
            self.frozen = copy.deepcopy(self.trunk)
            self.bit_head = copy.deepcopy(wm.det.pixel_decoder if method == "pixelseal" else wm.net.fc)
            for q in list(self.frozen.parameters()) + list(self.bit_head.parameters()):
                q.requires_grad_(False)
        self.lat = nn.ModuleList(nn.Conv2d(c, 64, 1) for _ in self.sizes for c in self.trunk.channels)
        # PixelSeal multiplies its residual by a perceptual masking map computed on the image (JND, Pixel Seal paper
        # eq. 10): strong on edges and dark areas, near zero on bright flat areas. Given as an input plane, it tells the
        # head where watermark evidence can exist at all.
        self.jnd = wm.model.attenuation if jnd and method == "pixelseal" else None
        cin = 64 * 4 * len(self.sizes) + extra + (3 if msg else 0) + (1 if self.jnd is not None else 0)
        block = lambda a, b: [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
        widths = [128, 64] if head == "small" else [256, 256, 128, 128, 64]      # big: more capacity against underfitting
        layers, prev = [], cin
        for w in widths:
            layers += block(prev, w); prev = w
        self.head = nn.Sequential(*layers, nn.Conv2d(prev, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if self.msg:                                               # the frozen extractor always runs in eval mode
            self.frozen.eval(); self.bit_head.eval()
        return self

    @torch.no_grad()
    def agreement(self, x, sign):
        """B x 3 x 128 x 128: local agreement with the message, its mean over the image, detection channel."""
        last = self.frozen(x)[-1]
        if self.method == "pixelseal":
            emb = self.bit_head.output_upscaling(last)
            lin = self.bit_head.linear
            pp = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]
            det, bits = pp[:, :1], pp[:, 1:]
        else:
            bits = torch.einsum("bchw,oc->bohw", last, self.bit_head.weight) + self.bit_head.bias[None, :, None, None]
            det = torch.zeros_like(bits[:, :1])
        agree = (bits * sign[:, :, None, None]).mean(1, keepdim=True)
        up = lambda t: F.interpolate(t.float(), size=(128, 128), mode="bilinear", align_corners=False)
        return torch.cat([up(agree), agree.mean((2, 3), keepdim=True).float().expand(-1, -1, 128, 128), up(det)], 1) / 5

    def forward(self, x, extra):                                   # x: B x 3 x 512 x 512 in [0, 1]; extra: B x 1 (+ nbits signs)
        ups, k = [], 0
        for size in self.sizes:
            xs = x if size == x.shape[-1] else F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
            if self.msg and size == self.sizes[0]:
                ups.append(self.agreement(xs, extra[:, 1:]).to(xs.dtype))
            for f in self.trunk(xs):
                ups.append(F.interpolate(self.lat[k](f), size=(128, 128), mode="bilinear", align_corners=False))
                k += 1
        planes = [extra[:, :1, None, None].expand(-1, -1, 128, 128)]
        if self.jnd is not None:
            with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
                planes.append(F.avg_pool2d(jnd_heatmap(self.jnd, x.float()), 4) * 20)
        return self.head(torch.cat([u.float() for u in ups] + planes, 1))[:, 0]


def message_signs(split, names):
    """B x nbits array of +1 / -1 for the given image names."""
    import json
    meta = {m["image_name"]: m["message"] for m in json.loads((Path("release") / split / "metadata.json").read_text())}
    return np.array([[1.0 if c == "1" else -1.0 for c in meta[n]] for n in names], np.float32)


class CopyPaste:
    """New training samples made of REAL watermarks (training fold only): clean blobs pasted onto real type A images
    (new holes), and blobs of real type A images pasted into clean images (new transplanted objects)."""
    def __init__(self, method, images, masks, full, tr, rng, n_blobs=300, n_carriers=400):
        from synth import blob
        self.rng, self.images, self.masks = rng, images, masks
        self.donors = [i for i in tr if full[i]]
        ds = [LocalisationDataset("release", s) for s in ("train", "test")]
        pool = [(d, i) for d in ds for i, m in enumerate(d.meta) if m["method"] != method]
        pick = rng.choice(len(pool), min(n_carriers, len(pool)), replace=False)
        self.carriers = [pool[k][0][pool[k][1]]["image"] for k in pick]           # preloaded once (speed)
        self.blobs = [blob(rng, rng.uniform(0.03, 0.3)).astype(bool) for _ in range(n_blobs)]

    def sample(self):
        r = self.rng
        d = self.donors[r.integers(len(self.donors))]
        clean = self.carriers[r.integers(len(self.carriers))]
        b = self.blobs[r.integers(len(self.blobs))]
        b = np.roll(b[:, ::-1] if r.random() < 0.5 else b, r.integers(-96, 97, 2), axis=(0, 1))
        b128 = b.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5
        if r.random() < 0.5:                       # new holes in a real watermarked image
            img = self.images[d].copy(); img[b] = clean[b]
            mask = self.masks[d] * (~b128)
            acc = 0.97
        else:                                      # a piece of a real watermarked image transplanted into a clean one
            img = clean.copy(); img[b] = self.images[d][b]
            mask = (self.masks[d] * b128).astype(np.uint8)
            acc = 0.53
        return img, mask.astype(np.uint8), np.array([acc], np.float32)


@torch.no_grad()
def predict(model, images, extra, device, batch=16, tta=False):
    """tta: average with the prediction of the mirrored image (PixelSeal / TrustMark decoding survives a flip)."""
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        x, e = batch_x(images[i:i + batch], device), torch.from_numpy(extra[i:i + batch]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            p = torch.sigmoid(model(x, e).float())
            if tta:
                p = (p + torch.sigmoid(model(x.flip(-1), e).float()).flip(-1)) / 2
        out.append(p.cpu().numpy())
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
    ap.add_argument("--head", default="small", choices=["small", "big"])
    ap.add_argument("--trunk-div", type=float, default=10, help="trunk learning rate = lr / this")
    ap.add_argument("--tag", default="", help="appended to the run name (distinguishes hyper-parameter variants)")
    ap.add_argument("--plots", action="store_true", help="write the two PNG plots (off by default: inode quota)")
    ap.add_argument("--msg", action="store_true", help="message-aware: agreement planes from a frozen extractor copy")
    ap.add_argument("--no-real", action="store_true", help="train on synthetic samples only (transfer gate), judge on ALL real images")
    ap.add_argument("--scales", default="1", help="e.g. 1,2: the trunk also sees the image at twice its working resolution")
    ap.add_argument("--copypaste", type=int, default=0, help="real-watermark copy-paste samples per epoch (training fold donors only)")
    ap.add_argument("--dil", action="store_true", help="dilated last trunk stage: deepest features on a grid twice as fine")
    ap.add_argument("--jnd", action="store_true", help="PixelSeal: the embedder's perceptual masking map as an input plane")
    ap.add_argument("--seed", type=int, default=0, help="another seed = another independent fit to average")
    ap.add_argument("--pseudo", default="", help="a submission file: its masks label the test images of the method, added to every epoch")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    from wmloc.watermarks import load_watermark
    print(f"device {a.device}, cuda available {torch.cuda.is_available()}", flush=True)
    images, masks, methods, extra, full, names = load_split("train", a.limit)
    sel = np.flatnonzero(methods == a.method)
    images, masks, methods, full, extra = images[sel], masks[sel], methods[sel], full[sel], extra[sel][:, -1:]
    names = [names[i] for i in sel]
    if a.msg:
        extra = np.concatenate([extra, message_signs("train", names)], 1)
    fold = folds_of(methods, full)
    tr = np.arange(len(images)) if a.fold == "all" else np.flatnonzero(fold != int(a.fold))
    ev = np.arange(len(images)) if a.fold == "all" else np.flatnonzero(fold == int(a.fold))
    if a.no_real:                                   # transfer gate: no real image in training, every real image judged
        tr, ev = np.array([], int), np.arange(len(images))
    name = f"ext_{a.method}_fold{a.fold}" + (f"_{a.synth_set}{a.synth}" if a.synth else "") + ("_synthonly" if a.no_real else "")
    print(f"{name}: train {len(tr)} real images, judged on {len(ev)}", flush=True)

    wm = load_watermark(a.method, device="cpu") if a.method != "stegastamp" else None
    model = ExtSegmenter(a.method, wm, scales=tuple(float(v) for v in a.scales.split(",")), msg=a.msg, head=a.head, dil=a.dil, jnd=a.jnd).to(a.device)
    trunk = list(model.trunk.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith("trunk.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": trunk, "lr": a.lr / a.trunk_div}, {"params": rest, "lr": a.lr}], weight_decay=1e-4)
    ps = None
    if a.pseudo:
        p_img, p_mask, p_extra = pseudo_set(a.pseudo, a.method)
        ps = (p_img, p_mask, p_extra[:, -1:])
        print(f"+ {len(p_img)} test images with the masks of {a.pseudo}", flush=True)
    steps = a.epochs * ((len(tr) + a.synth + a.copypaste + (len(ps[0]) if ps else 0) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr / a.trunk_div, a.lr], total_steps=steps, pct_start=0.1)
    if a.synth:
        z = load_synth(a.synth_set, a.method)
        s_img, s_mask = z["images"], z["masks128"]
        s_extra = (z["acc"] if "acc" in z else np.where(z["kinds"] == "full", 0.97, 0.53)).astype(np.float32)[:, None]   # full-image decode
        if a.msg:
            s_extra = np.concatenate([s_extra, z["bits"].astype(np.float32) * 2 - 1], 1)
        print(f"+ {a.synth} synthetic samples per epoch out of {len(s_img)}", flush=True)
    if a.copypaste:
        cp = CopyPaste(a.method, images, masks, full, tr, np.random.default_rng(1))
        name += f"_cp{a.copypaste}"
        print(f"+ {a.copypaste} copy-paste samples per epoch, {len(cp.carriers)} clean carriers, {len(cp.donors)} type A donors", flush=True)
    if a.msg:
        name += "_msg"
    if a.scales != "1":
        name += "_ms" + a.scales.replace(",", "-").replace(".", "p")
    if a.dil:
        name += "_dil"
    if a.jnd:
        name += "_jnd"
    if a.seed:
        name += f"_s{a.seed}"
    if a.tag:
        name += "_" + a.tag
    history, t0, probs = [], time.time(), None
    resume, first = OUT / f"resume_{name}.pt", 1                   # a job killed to give its GPU back restarts where it was
    if resume.exists():
        ck = torch.load(resume, map_location=a.device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        history, first = ck["history"], ck["epoch"] + 1
        rng = np.random.default_rng(1000 + first)
        print(f"resumed after epoch {ck['epoch']}", flush=True)
    for epoch in range(first, a.epochs + 1):
        model.train()
        order = [("real", i) for i in rng.permutation(tr)]
        if a.synth:
            order += [("synth", i) for i in rng.choice(len(s_img), a.synth, replace=False)]
        if a.copypaste:
            order += [("cp", i) for i in range(a.copypaste)]
        if ps:
            order += [("pseudo", i) for i in range(len(ps[0]))]
        order = [order[k] for k in rng.permutation(len(order))]
        losses = []
        for i in range(0, len(order), a.batch):
            chunk = order[i:i + a.batch]
            gen = {k: cp.sample() for k, (src, _) in enumerate(chunk) if src == "cp"} if a.copypaste else {}
            def pick(real, syn, field):
                out = []
                for k, (src, j) in enumerate(chunk):
                    out.append(real[j] if src == "real" else syn[j] if src == "synth" else ps[field][j] if src == "pseudo" else gen[k][field])
                return np.stack(out)
            x = batch_x(pick(images, s_img if a.synth else None, 0), a.device)
            y = torch.from_numpy(pick(masks, s_mask if a.synth else None, 1)).float().to(a.device)
            e = torch.from_numpy(pick(extra, s_extra if a.synth else None, 2)).to(a.device)
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
        if epoch < a.epochs and (epoch % 3 == 0 or not resume.exists()):
            try:                                                   # in place: no new inode once the file exists
                torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "history": history}, resume)
            except (OSError, RuntimeError) as e:
                print(f"resume not saved ({type(e).__name__}), training goes on", flush=True)
    resume.unlink(missing_ok=True)                                # frees an inode for the files below
    retry(lambda: torch.save(model.state_dict(), OUT / f"seg{name}.pt"))
    if a.fold != "all":
        retry(lambda: np.savez(OUT / f"oof_{name}.npz", names=np.array(names)[ev], probs=probs.astype(np.float16)))   # for post-processing CV
        if a.plots:
            plots(name, [(h[0], h[1], {m: h[2].get(m, (np.nan,) * 3) for m in ("pixelseal", "trustmark", "stegastamp")}) for h in history],
                  images[ev], masks[ev], probs, methods[ev], full[ev])
        return
    t_img, _, t_methods, t_extra, _, t_names = load_split("test")
    s = t_methods == a.method
    t_e = t_extra[s][:, -1:]
    if a.msg:
        t_e = np.concatenate([t_e, message_signs("test", [n for n, k in zip(t_names, s) if k])], 1)
    t_probs = predict(model, t_img[s], t_e, a.device).astype(np.float16)
    retry(lambda: np.savez(OUT / f"seg_probs_{a.method}_test.npz", names=np.array(t_names)[s], probs=t_probs))
    print("wrote", OUT / f"seg_probs_{a.method}_test.npz", flush=True)


if __name__ == "__main__":
    main()
