"""Mean token NLL of each endpoint's answers under exact fp16 base weights (HF transformers, official chat template).
Validated on the practice set (ref_likelihood.py): plain lowest, fp8 tied, LoRA 0.3 < 0.7 < 1.5 above.
  python ep_likelihood_hf.py MODEL FAMILY families.csv answers1.jsonl [answers2.jsonl ...]"""
import csv, json, sys, collections, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
name, family, famfile, files = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
fam = {r["endpoint"]: r["family"] for r in csv.DictReader(open(famfile))}
tok = AutoTokenizer.from_pretrained(name)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to(dev).eval()
by = collections.defaultdict(lambda: collections.defaultdict(list))
for f in files:
    for l in open(f):
        r = json.loads(l)
        if fam.get(r["endpoint"]) == family:
            by[r["id"] + "@" + f][r["endpoint"]].append((r["prompt"], r["response"]))
cent, raw = collections.defaultdict(list), collections.defaultdict(list)
with torch.no_grad():
    for key, d in by.items():
        vals = {}
        for e, items in d.items():
            for prompt, resp in items:
                pre = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                p_ids = tok(pre, add_special_tokens=False).input_ids
                a_ids = tok(resp, add_special_tokens=False).input_ids[:256]
                if not a_ids: continue
                ids = torch.tensor([p_ids + a_ids], device=dev)
                logits = model(ids).logits[0, len(p_ids) - 1:-1].float()
                lp = torch.log_softmax(logits, -1)[torch.arange(len(a_ids), device=dev), torch.tensor(a_ids, device=dev)]
                vals.setdefault(e, []).append(float(-lp.mean()))
        mu = np.mean([x for v in vals.values() for x in v])
        for e, v in vals.items():
            raw[e] += v; cent[e] += [x - mu for x in v]
print(name, family)
out = {}
for e in sorted(cent, key=lambda e: np.mean(cent[e])):
    c = np.array(cent[e]); out[e] = (float(c.mean()), float(c.std() / np.sqrt(len(c))))
    print(f"  {e} NLL {np.mean(raw[e]):.3f}  centred {c.mean():+.3f} (se {c.std()/np.sqrt(len(c)):.3f}) n={len(c)}")
json.dump(out, open(f"/private/tmp/claude-501/-Users-nathanbilliouw--claude-spares-spare-rvTGes/50d0f696-0e80-4bb0-9032-515a3450ad5b/scratchpad/hf_nll_{family}.json", "w"))
