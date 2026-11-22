"""How fast can one GPU train our two models? Random data, same step as training (forward, loss, backward, AdamW),
for several settings: number format of the autocast, channels-last memory, cudnn autotuner, batch size.
  python bench_speed.py --model real          (ImageNet ResNet-34, 512 input)
  python bench_speed.py --model ext --method pixelseal --scales 1,2"""
import argparse
import time

import numpy as np
import torch

from seg_train import loss_fn


def run(model, batch, dtype, cl, steps, extra_dim, scaler_on):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_on)
    img = np.random.randint(0, 255, (batch, 512, 512, 3), np.uint8)
    y = (torch.rand(batch, 128, 128, device="cuda") > 0.9).float()
    e = torch.rand(batch, extra_dim, device="cuda")
    def step():
        x = torch.from_numpy(img).to("cuda").permute(0, 3, 1, 2).float() / 255
        if cl:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            loss = loss_fn(model(x, e).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        return loss
    for _ in range(4):
        step()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(steps):
        loss = step()
    loss.item(); torch.cuda.synchronize()
    return batch * steps / (time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="real")
    ap.add_argument("--method", default="pixelseal")
    ap.add_argument("--scales", default="1")
    a = ap.parse_args()
    if a.model == "real":
        from seg_real import RealSegmenter
        make, extra_dim, batches = (lambda: RealSegmenter(pretrained=False)), 4, (8, 32)
    else:
        from seg_ext import ExtSegmenter
        from wmloc.watermarks import load_watermark
        wm = load_watermark(a.method, device="cpu")
        import copy
        make = lambda: ExtSegmenter(a.method, copy.deepcopy(wm), scales=tuple(float(v) for v in a.scales.split(",")))
        extra_dim, batches = 1, (16, 48)
    print(f"{a.model} {a.method if a.model == 'ext' else ''} scales {a.scales} on {torch.cuda.get_device_name(0)}", flush=True)
    for name, dtype, cl, bench, scaler in (("bfloat16 (current)", torch.bfloat16, False, False, False),
                                           ("float16 + scaler", torch.float16, False, False, True),
                                           ("float16 + channels-last", torch.float16, True, False, True),
                                           ("float16 + channels-last + cudnn autotune", torch.float16, True, True, True),
                                           ("bfloat16 + channels-last + cudnn autotune", torch.bfloat16, True, True, False),
                                           ("float32, no autocast", None, False, False, False)):
        torch.backends.cudnn.benchmark = bench
        for b in batches:
            model = make().to("cuda")
            if cl:
                model = model.to(memory_format=torch.channels_last)
            try:
                ips = run(model, b, dtype, cl, 30 if b <= 16 else 12, extra_dim, scaler)
                print(f"  {name:44s} batch {b:3d}: {ips:7.0f} images/s", flush=True)
            except Exception as ex:
                print(f"  {name:44s} batch {b:3d}: failed ({type(ex).__name__}: {str(ex)[:60]})", flush=True)
            del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
