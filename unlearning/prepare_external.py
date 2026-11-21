"""Download a public image set and save it as unlabeled 224x224 uint8 images, used as a retain anchor
(keep the student close to the original on images that are not forget images). External data is allowed for this task.

Default source: Imagenette 320px (fast.ai, ~13k photos of 10 ImageNet classes, ~340 MB).
Run ONCE on the LOGIN node (compute nodes have no internet), inside the unlearning data folder:
  cd $TEAM_DIR/unlearning && PYTHONPATH=$TEAM_DIR/code/$USER/unlearning .venv/bin/python \
      $TEAM_DIR/code/$USER/unlearning/prepare_external.py --n 8000
Writes external_images.npz (key "images", uint8 [N, 224, 224, 3]). The last --holdout images are kept out of training
by unlearn_distill.py and used to measure retain drift.
"""
import argparse
import io
import tarfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"


def to_224(data: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = im.size
    s = 224 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = im.size
    left, top = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((left, top, left + 224, top + 224)), dtype=np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=URL)
    p.add_argument("--archive", default="external/imagenette2-320.tgz")
    p.add_argument("--out", default="external_images.npz")
    p.add_argument("--n", type=int, default=8000, help="number of images kept (random subset)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    archive = Path(a.archive)
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {a.url} ...", flush=True)
        with requests.get(a.url, stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = archive.with_suffix(".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.rename(archive)
    print(f"reading {archive} ({archive.stat().st_size / 1e6:.0f} MB)", flush=True)

    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith((".jpeg", ".jpg", ".png"))]
        rng = np.random.default_rng(a.seed)
        members = [members[i] for i in rng.permutation(len(members))[: a.n]]
        images = []
        for i, m in enumerate(members):
            try:
                images.append(to_224(tar.extractfile(m).read()))
            except Exception as exc:  # skip unreadable files
                print(f"skip {m.name}: {exc}")
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(members)}", flush=True)
    arr = np.stack(images)
    np.savez(a.out, images=arr)
    print(f"wrote {a.out}: {arr.shape} {arr.dtype} ({Path(a.out).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
