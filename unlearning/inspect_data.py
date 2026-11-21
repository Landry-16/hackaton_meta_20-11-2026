"""Look at the forget set to identify the underlying dataset (is it a public one with a public test split?).

  python inspect_data.py [--per-class 4] [--classes 0 38 29 7 8 30]     (inside the unlearning data folder)
Writes out/forget_grid.png: one row per class (original's predicted class id at the left), and prints, per candidate
source resolution, how much detail survives a down/up-sampling round trip (near-zero error at 32 => upscaled 32x32,
i.e. CIFAR-like). Runs on CPU in about a minute, no GPU needed (login node is fine).
"""
import argparse
import struct
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from task_template import load_forget_images, load_model, predict


def write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def digit_strip(n: int, h: int) -> np.ndarray:
    """Tiny 3x5 bitmap font, scaled, to stamp the class id left of each row."""
    font = {"0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
            "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001001001001",
            "8": "111101111101111", "9": "111101111001111"}
    s, scale = str(n), 8
    out = np.full((h, 4 * 3 * scale, 3), 255, np.uint8)
    for k, ch in enumerate(s):
        bits = np.array([int(b) for b in font[ch]], np.uint8).reshape(5, 3)
        g = np.kron(bits, np.ones((scale, scale), np.uint8))
        y0, x0 = (h - g.shape[0]) // 2, k * 4 * scale
        out[y0:y0 + g.shape[0], x0:x0 + g.shape[1]][g == 1] = 0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=6)
    p.add_argument("--classes", type=int, nargs="*", default=None, help="default: the 10 most frequent")
    p.add_argument("--n", type=int, default=3500)
    a = p.parse_args()
    torch.set_num_threads(8)

    imgs = load_forget_images()[: a.n]
    x = imgs.permute(0, 3, 1, 2).float()
    print("resolution round-trip error (mean abs, 0-255) on 200 images:")
    for r in (32, 64, 96, 128, 160):
        sub = x[:200]
        rt = F.interpolate(F.interpolate(sub, size=r, mode="area"), size=224, mode="bilinear", align_corners=False)
        print(f"  {r:>3}px: {(rt - sub).abs().mean():.2f}")

    preds = predict(load_model(), imgs).argmax(1)
    counts = torch.bincount(preds, minlength=100)
    classes = a.classes or counts.argsort(descending=True)[:10].tolist()
    rows = []
    for c in classes:
        idx = (preds == c).nonzero().flatten()[: a.per_class].tolist()
        tiles = [imgs[i].numpy() for i in idx] + [np.full((224, 224, 3), 255, np.uint8)] * (a.per_class - len(idx))
        rows.append(np.concatenate([digit_strip(c, 224)] + tiles, 1))
        print(f"class {c}: {counts[c].item()} forget images")
    out = Path("out/forget_grid.png")
    out.parent.mkdir(exist_ok=True)
    write_png(out, np.concatenate(rows, 0))
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
