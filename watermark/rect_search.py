"""Type B images (a small watermarked area): find the rectangle whose crop decodes the known message best.

For every image:
  1. decode the full image: high bit accuracy means type A (almost all watermarked), so the mask is all ones;
  2. otherwise search rectangles, coarse grid then local refinement, scoring each crop by its bit accuracy
     against the known message; if the best crop clearly decodes, the mask is that rectangle, else all zeros.
Two thresholds per method (regime, detection) are chosen by 5-fold cross-validation on train with the official score.

Usage (run from the watermark folder, or through jureca/run.sh which runs it in the task data folder):
  quick local check:  python rect_search.py --split train --limit 9 --device cpu --stride 128
  full train run:     jureca/run.sh watermark watermark/rect_search.py --split train
  test submission:    jureca/run.sh watermark watermark/rect_search.py --split test
  plots only:         python rect_search.py --analyze
Outputs land in output/: rect_<split>.npz (search cache), plots/rect_*.png, submission_rect.pt (test)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wmloc.data import LocalisationDataset
from wmloc.utils import downsample_mask, miou_image, to_tensor

METHODS = ("pixelseal", "trustmark", "stegastamp")
RESIZE = {"pixelseal": (256, "bilinear"), "trustmark": (245, "bilinear"), "stegastamp": (400, "bicubic")}
MAX_ASPECT = {"pixelseal": 16, "stegastamp": 16, "trustmark": 2}      # from the task description
SKIP_SEARCH_ABOVE = 0.95   # shortcut: full images this well decoded are type A, no rectangle search; lower it if regime errors show up
OUT = Path("output")
SIZE = 512


# ---------------------------------------------------------------- search
def crop_scores(wm, method, img, boxes, sign, batch=128):
    """Bit accuracy (plus a tiny soft margin to break ties) of every crop box (y0, y1, x0, x1) against the message."""
    res, mode = RESIZE[method]
    out = []
    for i in range(0, len(boxes), batch):
        crops = [F.interpolate(img[:, :, y0:y1, x0:x1], size=(res, res), mode=mode, align_corners=False, antialias=True)
                 for y0, y1, x0, x1 in boxes[i:i + batch]]
        agree = wm.logits(torch.cat(crops).clamp(0, 1)).to(sign.device) * sign
        out.append((agree > 0).float().mean(1) + 1e-3 * torch.tanh(agree).mean(1))
    return torch.cat(out).cpu().numpy()


def valid(box, max_aspect, min_size):
    y0, y1, x0, x1 = box
    h, w = y1 - y0, x1 - x0
    return 0 <= y0 and y1 <= SIZE and 0 <= x0 and x1 <= SIZE and min(h, w) >= min_size and max(h, w) / min(h, w) <= max_aspect


def grid_boxes(stride, max_aspect):
    cuts = range(0, SIZE + 1, stride)
    return [b for b in ((y0, y1, x0, x1) for y0 in cuts for y1 in cuts for x0 in cuts for x1 in cuts) if valid(b, max_aspect, stride)]


def search(wm, method, img, sign, stride):
    """Coarse grid, then move each edge by halving steps while the score improves."""
    boxes = grid_boxes(stride, MAX_ASPECT[method])
    scores = crop_scores(wm, method, img, boxes, sign)
    best, best_score = boxes[int(scores.argmax())], float(scores.max())
    step = stride // 2
    while step >= 8:
        improved = True
        while improved:
            moves = [tuple(v + d if k == e else v for k, v in enumerate(best)) for e in range(4) for d in (-step, step)]
            moves = [m for m in moves if valid(m, MAX_ASPECT[method], 8)]
            if not moves:
                break
            s = crop_scores(wm, method, img, moves, sign)
            improved = float(s.max()) > best_score
            if improved:
                best, best_score = moves[int(s.argmax())], float(s.max())
        step //= 2
    return best, best_score


def run_search(split, limit, device, stride):
    from wmloc.watermarks import load_watermark
    print(f"device {device}, cuda available {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""), flush=True)
    wms = {m: load_watermark(m, device=device) for m in METHODS}
    ds = LocalisationDataset("release", split)
    n = min(limit or len(ds), len(ds))
    rows = {"names": [], "methods": [], "full_acc": [], "best_acc": [], "boxes": []}
    t0 = time.time()
    for i in range(n):
        it = ds[i]
        m, img = it["method"], to_tensor(it["image"]).to(device)
        sign = torch.tensor(it["message"].astype(np.float32) * 2 - 1, device=device)
        full = float((crop_scores(wms[m], m, img, [(0, SIZE, 0, SIZE)], sign)[0]))
        box, best = ((0, SIZE, 0, SIZE), full) if full >= SKIP_SEARCH_ABOVE else search(wms[m], m, img, sign, stride)
        for k, v in zip(rows, (it["image_name"], m, full, best, box)):
            rows[k].append(v)
        if n <= 25 or (i + 1) % 25 == 0 or i + 1 == n:
            print(f"{split} {i + 1}/{n} images ({m}, full acc {full:.2f}, best crop {best:.2f}), {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(exist_ok=True)
    tag = f"{split}_quick" if limit else split          # quick checks never overwrite a full run
    np.savez(OUT / f"rect_{tag}.npz", **{k: np.array(v) for k, v in rows.items()})
    return load_cache(tag)


def load_cache(tag):
    z = np.load(OUT / f"rect_{tag}.npz")
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- masks, tuning, CV
def auc_of(scores, labels):
    """Probability that a random positive scores above a random negative (0.5 = chance)."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def box_mask(box):
    m = np.zeros((SIZE, SIZE), np.uint8)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = 1
    return downsample_mask(m)


def candidates(cache):
    """The three possible masks per image: all zeros, the found rectangle, all ones."""
    ones, zeros = np.ones((128, 128), np.uint8), np.zeros((128, 128), np.uint8)
    return [(zeros, box_mask(b), ones) for b in cache["boxes"]]


def choose(full_acc, best_acc, thr_regime, thr_detect):
    """0 = zeros, 1 = rectangle, 2 = ones."""
    return np.where(full_acc >= thr_regime, 2, np.where(best_acc >= thr_detect, 1, 0))


REGIME_GRID, DETECT_GRID = np.arange(0.55, 0.97, 0.01), np.arange(0.50, 1.01, 0.01)


def tune(option_scores, full_acc, best_acc, use_rect=True):
    """Best (thr_regime, thr_detect) for the mean of per-image scores. option_scores: N x 3."""
    best = (-1, (0.90, 2.0))   # default when there is nothing to fit on (quick checks)
    for tr in REGIME_GRID:
        for td in (DETECT_GRID if use_rect else [2.0]):
            s = option_scores[np.arange(len(full_acc)), choose(full_acc, best_acc, tr, td)].mean()
            if s > best[0]:
                best = (s, (float(tr), float(td)))
    return best[1]


def folds_of(methods, full, k=5, seed=0):
    rng, fold = np.random.default_rng(seed), np.zeros(len(methods), int)
    for key in sorted(set(zip(methods, full))):         # sorted: same folds in every process (set order is random)
        idx = np.flatnonzero([(m, f) == key for m, f in zip(methods, full)])
        fold[rng.permutation(idx)] = np.arange(len(idx)) % k
    return fold


def cross_validate(cache, gts, full):
    """Held-out score per method for the regime baseline and the rectangle method, plus thresholds fit on all train."""
    opts = np.array([[miou_image(c, g) for c in cand] for cand, g in zip(candidates(cache), gts)])   # N x 3
    methods, fa, ba = cache["methods"], cache["full_acc"], cache["best_acc"]
    fold = folds_of(methods, full)
    report, params = {}, {}
    for m in METHODS:
        sel = methods == m
        for name, use_rect in (("regime", False), ("rect", True)):
            held = []
            for f in range(5):
                tr, te = sel & (fold != f), sel & (fold == f)
                if not te.any():
                    continue
                t = tune(opts[tr], fa[tr], ba[tr], use_rect)
                held.append(opts[te][np.arange(te.sum()), choose(fa[te], ba[te], *t)].mean())
            report[(m, name)] = (float(np.mean(held)) if held else float("nan"), float(np.std(held)) if held else 0.0)
        params[m] = tune(opts[sel], fa[sel], ba[sel], True)
    return report, params, opts


# ---------------------------------------------------------------- plots
def plots(cache, ds, report, opts, params, full):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    pdir = OUT / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    methods = cache["methods"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(METHODS))
    for j, (name, color) in enumerate((("regime", "#9aa5b1"), ("rect", "#2f80ed"))):
        mu = [report[(m, name)][0] for m in METHODS]
        sd = [report[(m, name)][1] for m in METHODS]
        bars = ax.bar(x + (j - 0.5) * 0.38, mu, 0.38, yerr=sd, capsize=4, color=color, label={"regime": "v1 regime only", "rect": "v2 + rectangle search"}[name])
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(x, METHODS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("5-fold held-out score (mean IoU)")
    ax.set_title("Watermark localisation: cross-validated score per method")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(pdir / "rect_cv_scores.png", dpi=130)

    typeb = ~full
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, c in zip(METHODS, ("#eb5757", "#27ae60", "#9b51e0")):
        s = typeb & (methods == m)
        ax.scatter(cache["best_acc"][s], opts[s, 1], s=12, alpha=0.7, color=c, label=m)
        ax.axvline(params[m][1], color=c, ls="--", lw=1)
    ax.set_xlabel("bit accuracy of the best crop (0.5 = chance)")
    ax.set_ylabel("score of the rectangle mask")
    ax.set_title("Type B images: does a well decoded crop mean a good rectangle?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdir / "rect_acc_vs_score.png", dpi=130)

    idx = [i for m in METHODS for i in np.flatnonzero(typeb & (methods == m))[:3]]
    fig, axes = plt.subplots(len(idx) // 3, 3, figsize=(10, 3.4 * (len(idx) // 3)), squeeze=False)
    for a, i in zip(axes.ravel(), idx):
        it = ds[int(i)]
        a.imshow(it["image"])
        a.contour(it["mask"], levels=[0.5], colors="lime", linewidths=1.5)
        y0, y1, x0, x1 = cache["boxes"][i]
        a.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=1.5))
        a.set_title(f"{it['method']}  crop acc {cache['best_acc'][i]:.2f}  score {opts[i, 1]:.2f}", fontsize=8)
        a.axis("off")
    fig.suptitle("green = true watermark, red = rectangle found", fontsize=10)
    fig.tight_layout()
    fig.savefig(pdir / "rect_gallery.png", dpi=110)
    print("plots in", pdir.resolve())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (quick check)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stride", type=int, default=64, help="coarse grid step in pixels")
    ap.add_argument("--analyze", action="store_true", help="reuse output/rect_train.npz, only redo CV and plots")
    a = ap.parse_args()

    if a.split == "test":
        params = json.loads((OUT / "rect_params.json").read_text())
        cache = run_search("test", a.limit, a.device, a.stride)
        cand = candidates(cache)
        pick = [choose(np.array([fa]), np.array([ba]), *params[m])[0] for fa, ba, m in zip(cache["full_acc"], cache["best_acc"], cache["methods"])]
        masks = np.stack([c[p] for c, p in zip(cand, pick)])
        names = [str(n) for n in cache["names"]]
        if not a.limit:
            assert masks.shape == (1020, 128, 128) and len(set(names)) == 1020
        torch.save({"image_names": names, "masks": torch.from_numpy(masks).bool()}, OUT / "submission_rect.pt")
        print("test choices (zeros, rectangle, ones):", np.bincount(pick, minlength=3).tolist(), "-> wrote", OUT / "submission_rect.pt")
        return

    cache = load_cache("train") if a.analyze else run_search("train", a.limit, a.device, a.stride)
    ds = LocalisationDataset("release", "train")
    by_name = {ds.meta[i]["image_name"]: i for i in range(len(ds))}
    order = [by_name[str(n)] for n in cache["names"]]
    gts = [ds.m128[str(n)] for n in cache["names"]]
    full = np.array([bool(ds.meta[i]["full_watermark"]) for i in order])
    report, params, opts = cross_validate(cache, gts, full)
    print("\n5-fold held-out score (mean +- std over folds)")
    for m in METHODS:
        print(f"  {m:11s} regime only {report[(m, 'regime')][0]:.4f} +- {report[(m, 'regime')][1]:.3f}   "
              f"+ rectangle {report[(m, 'rect')][0]:.4f} +- {report[(m, 'rect')][1]:.3f}")
    for name in ("regime", "rect"):
        print(f"  overall {name}: {np.mean([report[(m, name)][0] for m in METHODS]):.4f}")
    print("thresholds (regime, detection) fit on all train:", params)
    if not a.limit:
        (OUT / "rect_params.json").write_text(json.dumps(params))
    plots(cache, _Ordered(ds, order), report, opts, params, full)


class _Ordered:
    """Index the dataset in cache order."""
    def __init__(self, ds, order):
        self.ds, self.order = ds, order

    def __getitem__(self, i):
        return self.ds[self.order[i]]


if __name__ == "__main__":
    main()
