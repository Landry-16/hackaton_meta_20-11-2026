"""Contact sheet of an images npz (e.g. invert.py output), CPU only, to check that generated anchor images look natural.

  python grid_npz.py external_inv.npz --per-class 40 --rows 8 --cols 8 --out ~/inv_grid.png
invert.py writes classes in blocks of --per-class images: row r shows the first --cols images of the r-th class block.
Write the PNG outside /p/scratch (e.g. your home) so it costs no scratch inode.
"""
import argparse
from pathlib import Path

import numpy as np

from inspect_data import write_png


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=Path)
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--size", type=int, default=112, help="tile size in pixels (downsampled by striding)")
    p.add_argument("--out", type=Path, default=Path.home() / "inv_grid.png")
    a = p.parse_args()
    imgs = np.load(a.npz)["images"]
    step = max(224 // a.size, 1)
    rows = []
    for r in range(a.rows):
        start = r * a.per_class
        if start >= len(imgs):
            break
        tiles = [imgs[start + c][::step, ::step] for c in range(min(a.cols, len(imgs) - start))]
        tiles += [np.full_like(tiles[0], 255)] * (a.cols - len(tiles))
        rows.append(np.concatenate(tiles, 1))
    sheet = np.concatenate(rows, 0)
    write_png(a.out.expanduser(), np.ascontiguousarray(sheet))
    print(f"{len(imgs)} images in {a.npz}; wrote {a.out} ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
