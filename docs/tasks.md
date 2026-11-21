# The three tasks

Server: `http://34.147.229.55`, header `X-API-Key: $HACK_API_KEY`. Leaderboard: http://34.147.229.55/leaderboard_page

Overall ranking: each task score rescaled 0 to 1 across teams (best = 1, lowest = 0), averaged over the 3 tasks. **No score on one task = not ranked.** Public board = ~30 % of the evaluation data, final = the other 70 %.

## Task 1: Watermark localisation (`watermark/`, task id `34-watermark-localisation-image`)
- 1,530 PNG images, 512x512, watermarked in a rectangular sub-region with PixelSeal (256 bits), TrustMark (100 bits, variant Q) or StegaStamp (100 bits). Then edited: outpainting, object removal, object addition, object transplantation.
- Train: 510 images (170 per method) with mask (512 and 128), method, message, `full_watermark`. Test: 1,020 images with method and message (no mask).
- Predict a 128x128 grid per image (cell = 4x4 px, 1 if at least 50 % watermarked).
- Score: per image mean of IoU(clean) and IoU(watermarked), then mean over images. Empty union counts 1.
- Submission: `torch.save({"image_names": names, "masks": bool[1020,128,128]}, "submission.pt")`, at most 20 MB.
- Starter kit: `wmloc/watermarks.py` gives `embed`, `logits`, `decode`, `features` (per-position logits for PixelSeal and TrustMark). `wmloc/utils.py` has the official downsampling and score.
- Ideas: we know the embedded message, so slide a window (or use per-position logits) and mark cells where bit accuracy against the known message is high; one strategy per method; generate extra training data with `embed` + synthetic edits and train a small segmentation head on extractor features.

## Task 2: Model equality (`model_equality/`, task id `33-model-equality`)
- 33 endpoints `ep-01..ep-33`. Same model = same checkpoint, quantization and LoRA. Watermark, system prompt, persona or decoding strategy do NOT change identity.
- Score: 0.1 x MCC(same family, all 528 pairs) + 0.9 x MCC(same model, pairs inside a family).
- Budget: 100 answers per scored endpoint for the whole event, 256 output tokens, 1,024 input tokens, 4 requests at a time, temperature and seed ignored, system messages rejected.
- Practice: `ref-01..ref-09` unlimited, answers in `practice_answers.csv` (plain, fp8 quantized, system prompt, persona, watermark, temperature 1.0, LoRA at 0.3 / 0.7 / 1.5). Score locally with `python score_practice.py my.csv`.
- Submission: CSV `endpoint,family,model`, one line per `ep-`, one submission per 30 min.
- Ideas: MMD with a string kernel (Gao et al.), rank-based uniformity test (RUT); prompts that force near-deterministic outputs (exact recall, arithmetic, completion of a fixed text) to expose quantization and LoRA shifts while neutralising persona and system prompt effects; cluster on pairwise distances, tune the threshold on `ref-*`.

## Task 3: Unlearning (`unlearning/`, task id `31-unlearning`)
- ViT-B/16 (timm `vit_base_patch16_224`, 100 classes, LayerScale 1e-4) in `model.pt`; 3,500 forget images `forget_images.npz` (uint8 [N,224,224,3], no labels). Preprocess: `(x/255 - 0.5)/0.5`, use `preprocess()` from the template.
- Goal: behave like gold models retrained without the forget set, on the forget images and on hidden altered copies of them.
- Forget score = mean/std loss score x prediction score (KL to the gold average, softmax T=2). Retain score = accuracy ratio on a control set x 1/(1 + KL(orig || sub)). Final = forget x retain. **Gate: below 70 % held-out accuracy = 0.**
- Submission: float32 state dict, `strict=True`, at most 450 MB. Server runs eval mode with fp16 autocast.
- Ideas: SCRUB (Kurmanji et al.); gradient ascent alone is penalised; distill the original model on augmented forget images towards a "never seen" target (e.g. soften the predicted class, keep the runner-up classes); SalUn weight saliency; keep retain KL low by limiting how many layers move (head + last blocks only).
