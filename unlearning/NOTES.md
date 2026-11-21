# Unlearning notes (task 31)

Source: Kurmanji et al., "Towards Unbounded Machine Unlearning" (SCRUB, NeurIPS 2023, arXiv 2302.09880),
read against our scoring (docs/tasks.md, organizer PDF). Owner: @Mirado-77.

## Our setting vs the paper
| | Paper | Us |
|---|---|---|
| Model | ResNet-18 / All-CNN, CIFAR / Lacuna | ViT-B/16, 100 classes |
| Forget set | labelled | 3,500 images, **no labels** |
| Retain set D_r | available, used in every min-step | **not available** |
| Goal | depends on application | "User Privacy": match retrain-from-scratch (gold), not maximal error |
| Measure | errors, MIA | loss mean/std vs gold, KL(p_gold‖p_sub) at T=2, on exact + altered images; retain = acc ratio × 1/(1+KL(orig‖sub)) on control |

Our score is exactly the paper's UP scenario: **too much forgetting is as bad as too little**.
Gold models still classify most forget images correctly (as unseen test images), with less confidence.

## SCRUB in one paragraph
Student w_u initialised from teacher w_o (frozen original). d(x) = KL(p_teacher(x) ‖ p_student(x)).
- max-step (one epoch on D_f): maximise d on forget images (move away from teacher).
- min-step (one epoch on D_r): minimise α·d + γ·CE on retain images (stay close to teacher).
- Loop: `for i in range(STEPS): if i < MAX_STEPS: max_epoch(); min_epoch()`, i.e. alternate, then a few extra min-only epochs at the end.
- **SCRUB+R (rewind):** save a checkpoint every epoch, measure error on a validation set of the same distribution as the forget set, pick the checkpoint whose forget error is closest to it. This is the variant that matches retrain (MIA ≈ 50 %).

Hyper-parameters reported (Table 3, Sec. 9):
- lr 5e-4, decayed ×0.1 after some epochs (decay is "crucial" against oscillation).
- Adam for small scale; SGD momentum 0.9 or Adam at large scale; weight decay 5e-4 (large) / 0.1 (small).
- MAX_STEPS 2–5, STEPS (min) 3–10. Forget batch 8–512, retain batch 32–256.
- α, γ in [0.1, 3]: results insensitive (Fig. 5). CE term gives small extra retain protection (Fig. 10).

Ablations (Figs. 6–9):
- max-steps only (pure ascent on KL) wrecks retain and validation error: never do it alone.
- too few max-steps → forget error falls back; too many → oscillation.
- the final min-only epochs are needed to restore utility.

Baselines worth knowing: NegGrad+ (β·CE_retain − (1−β)·CE_forget, β≈0.95–0.9999) is a solid second; finetune / CF-k do not forget; Bad-T (distill to a random "bad teacher") forgets poorly and hurts utility.

## What transfers, what does not
1. **No D_r** → the min-step needs a substitute that keeps us close to the original on the hidden control set:
   - L2 / KL anchoring to the original weights; train only head + last blocks (fewer moved weights = lower retain KL).
   - Distill teacher on proxy inputs: strong augmentations, mixup/cutmix between forget images, noise. Caveat: augmented forget images resemble the hidden "altered copies", which belong on the forget side.
2. **No labels** → the original almost surely fits the forget set, so `argmax p_orig` is a good pseudo-label for CE.
3. **Rewind target without gold:** gold forget behaviour ≈ the original's behaviour on unseen images of the same distribution. We have none locally, so use (a) the public leaderboard as the rewind signal, (b) a target accuracy / loss level we sweep (e.g. forget acc ∈ {gate-ish … val-ish}), and pick by public score with few submissions.
4. **Prediction score needs the right wrong class**: pushing away from the teacher (SCRUB max-step) gives arbitrary wrong classes. Better aligned target: a softened version of the original's own distribution (its runner-up classes are what a gold model would pick), e.g. distill towards p_orig at high T, or p_orig averaged over strong augmentations, with the top class damped.
5. Scoring runs fp16 autocast, eval mode: evaluate locally the same way (`predict()` in task_template.py).

## Plan
1. Proxy metrics script: on forget images, loss mean/std w.r.t. pseudo-labels, accuracy, KL(orig‖sub); drift on augmented / noise inputs as a retain proxy.
2. Method A: soft-target distillation on last k blocks + head, L2 anchor, checkpoints along the way (rewind).
3. Method B: SCRUB-style alternating max/min with a proxy retain set, then rewind.
4. Submit the most conservative checkpoint first (stay above the 70 % gate), post every submission on the board.
