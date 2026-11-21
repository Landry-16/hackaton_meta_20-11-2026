"""All-zeros masks for every watermark test image: a valid first submission, no model needed.
Usage: python baselines/watermark_zeros.py data/test.zip data/watermark_zeros.pt"""
import json
import sys
import zipfile

import torch

with zipfile.ZipFile(sys.argv[1]) as z:
    meta = next(n for n in z.namelist() if n.endswith("metadata.json"))
    names = [m["image_name"] for m in json.loads(z.read(meta))]

assert len(names) == len(set(names)) == 1020, f"expected 1020 unique test images, got {len(names)}"
torch.save({"image_names": names, "masks": torch.zeros(len(names), 128, 128, dtype=torch.bool)}, sys.argv[2])
print("wrote", sys.argv[2], len(names), "masks")
