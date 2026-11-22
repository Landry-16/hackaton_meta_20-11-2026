"""Send a submission file from a LOGIN node (compute nodes have no internet; login nodes may have no /p/scratch).
Same request as watermark/submission_template.py, without torch. The key is read from HACK_API_KEY (~/.bashrc).
  rsync the file to ~/sub/ on the login node, then:  python3 ~/submit_login.py ~/sub/submission_vN.pt"""
import os
import sys

import requests

BASE_URL, TASK_ID = "http://34.147.229.55", "34-watermark-localisation-image"
path = sys.argv[1]
with open(path, "rb") as f:
    resp = requests.post(f"{BASE_URL}/submit/{TASK_ID}", headers={"X-API-Key": os.environ["HACK_API_KEY"]},
                         files={"file": (os.path.basename(path), f, "application/octet-stream")}, timeout=(10, 120))
try:
    print(resp.status_code, resp.json())
except Exception:
    print(resp.status_code, resp.text[:300])
