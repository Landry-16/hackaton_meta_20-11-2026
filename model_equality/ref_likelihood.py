"""Validate plain identification on the practice endpoints: NLL of each ref's answers under SmolLM2 base weights
(the refs leaked 'SmolLM, trained by Hugging Face'). Truth: ref-01 plain, ref-02 fp8, ref-04/07/09 LoRA."""
import json, sys, glob, collections, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
name = sys.argv[1]
tok = AutoTokenizer.from_pretrained(name)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(dev).eval()
files = sys.argv[2:] or sorted(glob.glob("*reference_answers.jsonl"))
by = collections.defaultdict(lambda: collections.defaultdict(list))
for f in files:
    for l in open(f):
        r = json.loads(l)
        by[r["id"] + "@" + f][r["endpoint"]].append((r["prompt"], r["response"]))
cent = collections.defaultdict(list); raw = collections.defaultdict(list)
with torch.no_grad():
    for key, d in by.items():
        vals = {}
        for e, items in d.items():
            for prompt, resp in items:
                pre = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                p_ids = tok(pre, add_special_tokens=False).input_ids
                a_ids = tok(resp, add_special_tokens=False).input_ids[:96]
                if not a_ids: continue
                ids = torch.tensor([p_ids + a_ids], device=dev)
                logits = model(ids).logits[0, len(p_ids) - 1:-1].float()
                lp = torch.log_softmax(logits, -1)[torch.arange(len(a_ids)), torch.tensor(a_ids, device=dev)]
                vals.setdefault(e, []).append(float(-lp.mean()))
        mu = np.mean([x for v in vals.values() for x in v])
        for e, v in vals.items():
            raw[e] += v; cent[e] += [x - mu for x in v]
kinds = {"ref-01":"plain","ref-02":"fp8","ref-03":"sysprompt","ref-04":"lora0.7","ref-05":"persona","ref-06":"watermark","ref-07":"lora1.5","ref-08":"temp1.0","ref-09":"lora0.3"}
print(name)
for e in sorted(cent, key=lambda e: np.mean(cent[e])):
    c = np.array(cent[e]); print(f"  {e} {kinds[e]:10s} NLL {np.mean(raw[e]):.3f}  centred {c.mean():+.3f} (se {c.std()/np.sqrt(len(c)):.3f}) n={len(c)}")
