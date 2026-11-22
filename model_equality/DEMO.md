# Task 2 demo (2 minutes)

**Hook (15 s).** 33 API endpoints, 3 model families. Which endpoints serve the *same* weights? The catch: some
endpoints are the same model in disguise (a hidden system prompt, a persona, a watermark, a different
temperature), and every statistical test we tried says the disguised ones are *different*.

**What failed (20 s).** Two-sample tests on outputs (MMD with string and sentence-embedding kernels, behavioural
features, modal answers): on the practice set a watermark looks farther from its own base model than a LoRA at
0.7. Our first submissions scored 0.10.

**The idea (40 s).** Stop comparing distributions; recognise each disguise by its own fingerprint.
- *System prompt / persona*: ask the endpoint to repeat its instructions. A plain Qwen prints its default
  "You are Qwen, created by Alibaba Cloud"; disguised ones print a romantic-writer persona, a "Pinescript Genie",
  "use short paragraphs...".
- *Watermark*: near-deterministic answers.
- *Temperature*: fit the sampling temperature under the open fp16 base weights: the plain endpoint samples at
  T = 0.7, the disguised one at exactly T = 1.0.
Leaks and likelihoods also identify the base models: Qwen2.5-3B, Qwen2.5-1.5B, Llama-3.2-1B.

**The last bit (25 s).** The plain endpoint versus its fp8 copy: a likelihood ratio between the fp16 weights and
simulated fp8 weights (validated on the practice set: plain +42, fp8 -33), then one public probe per candidate:
the score rises only if the candidate is the true plain.

**Result (20 s).** 0.10 -> 0.55 (disguises) -> 0.70 (F2 plain) -> 0.85 (temperature fit fixes F1), with about
60 of 100 answers spent per endpoint. Every change is a fact about one endpoint, not a tuned threshold, so it
carries over to the hidden 70 % of pairs.

Numbers to have on screen: the submission table in METHOD.md, and one leak example per family.
