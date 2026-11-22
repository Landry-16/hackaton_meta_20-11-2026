"""Find the plain endpoint of a family: mean token NLL of each endpoint's answers under the base weights.

Why this is fair now (it was not for Codex's likelihood filter): the candidates compared here are the endpoints
left once the four disguises are removed (plain, quantized and LoRA variants), and they all use the same default
decoding, so the decoding confound is gone. The plain (and an fp8 copy) should score the lowest NLL under the base
weights; LoRA variants score higher. The chat templates match what the endpoints leaked: Qwen's default system
prompt, and Llama 3's 'Cutting Knowledge Date' header.

  python plain_likelihood.py --family F2 --model qwen2.5:1.5b-instruct-fp16 answers.jsonl completion.jsonl ...
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from llama_cpp import Llama

CACHE = Path.home() / ".ollama/models"
QWEN_SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def blob(tag):
    name, ver = tag.split(":")
    manifest = json.loads((CACHE / "manifests/registry.ollama.ai/library" / name / ver).read_text())
    layer = next(x for x in manifest["layers"] if x["mediaType"].endswith(".model"))
    return str(CACHE / "blobs" / layer["digest"].replace(":", "-"))


def prefix(tag, question):
    if tag.startswith("llama"):
        return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n" + question +
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    return (f"<|im_start|>system\n{QWEN_SYS}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n")


def nll(llm, pre, response, cap=96):
    p = llm.tokenize(pre.encode(), add_bos=False, special=True)
    a = llm.tokenize(response.encode(), add_bos=False, special=False)[:cap]
    if not a:
        return None
    llm.reset()
    llm.eval(p + a)
    logits = np.array(llm.scores[len(p) - 1: len(p) + len(a) - 1])
    m = logits.max(1)
    logz = m + np.log(np.exp(logits - m[:, None]).sum(1))
    return float((logz - logits[np.arange(len(a)), a]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers", nargs="+")
    ap.add_argument("--families", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--model", required=True, help="ollama tag, e.g. qwen2.5:1.5b-instruct-fp16")
    ap.add_argument("--ids", help="comma-separated prompt ids to keep (default: all)")
    ap.add_argument("--out")
    a = ap.parse_args()
    fam = {r["endpoint"]: r["family"] for r in csv.DictReader(open(a.families))}
    keep = set(a.ids.split(",")) if a.ids else None
    rows = []
    for path in a.answers:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if fam.get(r["endpoint"]) == a.family and (keep is None or r["id"] in keep):
                    rows.append(r)
    llm = Llama(model_path=blob(a.model), n_ctx=1024, logits_all=True, verbose=False, n_gpu_layers=-1)
    per = defaultdict(list)
    by_prompt = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = nll(llm, prefix(a.model, r["prompt"]), r["response"])
        if v is not None:
            per[r["endpoint"]].append(v)
            by_prompt[r["id"]][r["endpoint"]].append(v)
    # prompt-centred NLL: removes prompt difficulty, so endpoints are compared on the same footing
    centred = defaultdict(list)
    for p, d in by_prompt.items():
        mu = np.mean([x for v in d.values() for x in v])
        for e, v in d.items():
            centred[e] += [x - mu for x in v]
    print(f"family {a.family}, model {a.model}, {len(rows)} answers")
    out = []
    for e in sorted(per, key=lambda e: np.mean(centred[e])):
        c = np.array(centred[e])
        out.append((e, float(np.mean(per[e])), float(c.mean()), float(c.std() / np.sqrt(len(c))), len(c)))
        print(f"  {e}  mean NLL {np.mean(per[e]):.3f}  centred {c.mean():+.3f} (se {c.std() / np.sqrt(len(c)):.3f})  n={len(c)}")
    if a.out:
        json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
