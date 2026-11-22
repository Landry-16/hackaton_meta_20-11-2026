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

## Status (21 Nov, 18:30) - read this first when resuming
Code (branch `claude/dazzling-heisenberg-cdsx58`): `unlearn_distill.py` (method A), `proxy_metrics.py`, `ulib.py`,
`submit_queue.py` (auto-submits `logs/queue.txt`, one per ~6 min cooldown, scores in `logs/submissions.csv`).

Facts from JURECA:
- Forget set: 45 classes, ~130 images each (selective unlearning, not class unlearning). Original fits it perfectly
  (clean loss 0.0003, acc 1.0; 0.988 under augmentation).
- An epoch of distillation takes ~8 s on one A100: a 25-30 epoch run is ~5 min. JURECA queue is the bottleneck
  (use `sbatch -J hold jureca/hold.sh` and run inside it, see jureca/run.sh).
- Server cooldown between unlearning submissions: ~360 s.

Public scores (forget-set proxies at the submitted epoch, pseudo-labels = original argmax):
| checkpoint | acc | loss mean / std | public |
|---|---|---|---|
| original | 1.00 | 0.0003 / 0.002 | 0.0 |
| tc0.3 lr2e-5 e4 | 1.00 | 0.015 / 0.08 | 0.0016 |
| tc0.5 lr1e-4 e30 | 0.99 | 0.18 / 0.15 | 0.0108 |
| tc0.3 lr5e-5 e30 | 0.79 | 0.50 / 0.43 | 0.0308 |

Reading: more forgetting = higher score so far; gold loss on the forget set is probably well above 0.5.
Per-image caps (`--conf-min/--conf-max`, harder images under strong augmentation get lower caps) give a loss std
larger than the mean, closer to a never-seen model. Queued / next: cr0.1-0.8 e25 (loss 0.44, std 0.63), then much
stronger caps (cr0.05-0.5, cr0.03-0.35, cr0.02-0.2) to bracket the gold level. Watch the 70 % gate (score 0).

JURECA commands (clone at `$TEAM_DIR/code/ravelonahina1`, data/checkpoints at `$TEAM_DIR/unlearning/out/`):
- run: `sbatch -J unl-X jureca/job.sh unlearning unlearning/unlearn_distill.py --conf-min 0.05 --conf-max 0.5 --lr 1e-4 --epochs 25 --save-every 5`
- queue a checkpoint: `echo $TEAM_DIR/unlearning/out/<file>.pt >> logs/queue.txt`
- submitter (login node): `nohup $TEAM_DIR/unlearning/.venv/bin/python unlearning/submit_queue.py > logs/submit_queue.out 2>&1 &`
- scores: `column -s, -t < logs/submissions.csv`

## Literature cross-check (21 Nov evening)
Our setting = no retain set, no labels, match retrained models. Closest work:
- RKLU (ESANN 2026, retain-free): fine-tune with KL to a *modified* target distribution on the forget set only.
  Same family as our method A (soft-target distillation) -> independent support for the approach.
- LFSSD, Loss-Free Selective Synaptic Dampening (Foster et al., ICLR 2024 tiny, arXiv 2402.19308; SSD AAAI 2024,
  arXiv 2308.07707): no training, no labels. Importance of each weight = gradient of ||output||_2 on forget images vs
  on general data; weights much more important for the forget set get dampened. Seconds of GPU. Candidate method B,
  or a warm start before distillation; general-data importance can come from external_images.npz.
- Learning to Unlearn, instance-wise (Cha et al., AAAI 2024, arXiv 2301.11578): forget set only; adversarial examples
  of forget points as retain proxy + continual-learning weight-importance regulariser.
- Label-Agnostic Forgetting (Shen et al., ICLR 2024, arXiv 2404.00506): label-free; contrastive loss keeps
  representations of remaining data aligned with the original.
- NeurIPS 2023 unlearning competition findings (Triantafillou et al., arXiv 2406.09073): evaluation also compares
  output distributions to retrained models; top entries combine re-initialisation / noise of some layers with
  distillation on the retain set and forget-remember cycles (they had a retain set, we use external images instead).

## Status (21 Nov, 22:10) - supersedes the 18:30 table
Best public: **0.1869** = `unlearn_distill.py --conf-min 0.002 --conf-max 0.02 --external external_images.npz --lr 1e-4 --epochs 20`
(forget acc 0.14, loss 2.80/1.63, held-out external agreement 0.82).

| run (all with Imagenette anchor unless noted) | forget loss | ext agree | public |
|---|---|---|---|
| caps 0.05-0.5, no anchor | 0.85 | - | 0.0668 |
| caps 0.05-0.5 | 0.93 | 0.85 | 0.0749 |
| caps 0.03-0.35 | 1.35 | 0.84 | 0.1037 |
| caps 0.02-0.2 | 1.96 | 0.84 | 0.1393 |
| caps 0.01-0.1 | 2.29 | - | 0.1559 |
| caps 0.005-0.05 | 2.64 | - | 0.1779 |
| caps 0.002-0.02 | 2.80 | 0.82 | **0.1869** |
| caps 0.02-0.2 + --other-tau 3 (spread wrong mass) | 0.73 (acc 0.98) | 0.85 | 0.0152 |

Learned: (1) the external anchor helps at equal forgetting; (2) score rises monotonically with forgetting so far;
(3) gold models put their mass on the original's runner-up (sibling) classes: spreading it kills the score;
(4) caps 0.001-0.01 are unstable (less forgetting, agreement 0.80): 0.002-0.02 is the floor.
Rules now: ONE node per team, GPU only via `jureca/gpu_submit.sh`, max 2 unlearning jobs queued (Mirado + Philibert).
Submitter: `submit_queue.py --queue logs/queue_v2.txt --log logs/submissions_v2.csv` (single-instance lock).
Never submit a checkpoint whose external agreement is below ~0.80 (70 % validation gate).

## Status (21 Nov, 23:40) - corrections from Philibert's review
Best public: **0.2706** = caps 0.002-0.02 + `--external external_images.npz`, `--lr 2e-4 --epochs 35`
(forget loss 5.25/1.56, acc 0.00; augmented loss 3.84, acc 0.105; ext agree 0.83). lr 1e-4 x 35 ep: 0.2505.
Dead ends (comparable runs): weight soup of the two best (0.2446), lr 3e-4 x 20 (agree 0.81, less forgetting),
LFSSD alone (drifts before it forgets), LFSSD as --init (same forgetting, agree 0.795).
Caveats on earlier conclusions (runs NOT comparable, treat as open questions):
- caps 0.001-0.01 "unstable": that run was killed by the disk error and judged on an earlier epoch.
- --train-blocks 8: never finished (killed twice).
- --other-tau 3 (0.0152): also far less forgetting (acc 0.98, loss 0.73), so "spreading kills the score" is
  confounded with weak forgetting.
- Scores of the old e20 checkpoints (0.1869, 0.2402) point to files later overwritten by e20 of 35-epoch runs.
Fixed bugs: --seed had no effect and evaluation reset the training RNG (fixed 736663f); seed and total epochs are now
in checkpoint names; the submitter skips files removed during its cooldown (e4bc00b).
Open tests: --clean-frac 0.2 (augmented copies forgotten less than clean ones), --alpha 3 (retain weight),
seed 1 of the best recipe (Philibert, running), lr 2e-4 x 50 ep (Mirado, running).

## Process from 21 Nov 23:55 (agreed by Mirado): rounds
Philibert's Claude writes each round's plan (max 2 experiments, one per GPU slot, one change vs the reference recipe)
and its conclusions; Mirado's Claude runs, keeps this file and the results table, and is the only submitter.
Nothing is launched outside a plan; new ideas go to the next round. Mirado can overrule; if no answer within 20 min,
Mirado's Claude continues from the last validated plan.
Reference recipe: caps 0.002-0.02 + --external external_images.npz, lr 2e-4 x 35 ep (public 0.2706; seed 1 locally
identical: clean loss 5.21, aug 3.96, ext agree 0.824). lr 2e-4 x 50 ep overshoots (0.2599).
Round 1: --clean-frac 0.2 (Mirado) and --alpha 3 (Philibert), both at 35 ep.
Result format: command | clean loss (std), acc | aug loss, acc | ext agree, ext KL | public score if submitted.

## Round 1 (plan by Philibert, 23:55)
- Best public **0.304** = 0.2706 checkpoint with head scale 1.25 (`soup.py <ckpt> --head-scales 1.25`):
  local forget loss 6.15 (2.04), ext KL 0.129. Sharper outputs help.
- Withdrawn: "forgetting peaks at clean loss 5-5.5". Supported: longer training (50 ep, 0.2599) did not help,
  sharper outputs did. No gap under ~0.01 counts until seed 1's public score gives the seed-to-seed gap.
- Submission order: head scale 1.5 (if > 0.304 then 2.0, else keep 1.25 and stop), seed 1 raw, A (--clean-frac 0.2),
  B (--alpha 3), init-LFSSD e20.
- Diagnostic C (Philibert): is forget_images.npz class-sorted? All local metrics use the first 1000 images; if
  sorted they cover ~8 classes and are biased -> switch --eval-n to a random subset.

## Round 2 results (22 Nov, 00:50)
- Seed gap: seed 1 raw 0.2701 vs seed 0 raw 0.2706 -> gaps >= ~0.005 are real.
- Head scale on the 0.2706 checkpoint: 1.25 = **0.304** (best), 1.5 = 0.2786 -> scan stopped at 1.25.
- A raw (--clean-frac 0.2, 35 ep) = 0.2853 vs 0.2706: real gain (augmented copies forgotten closer to clean ones).
- init-LFSSD e20 = 0.2411 vs 0.2402 (ext agree 0.795 vs 0.83): small agreement drops cost nothing measurable.
- mask45 (head.bias -10 on the 45 forget classes) = **0.0** (utility gate): the 45 classes are NOT removed;
  gold models still predict them; the original's runner-up stays the right target even when it is a forget class.
- B raw (--alpha 3) dropped from the queue: weaker forgetting (loss 3.86) mixed with better retain (KL 0.067).
- Server cooldown is now 10 min (6 submissions/hour); submitter runs with --cooldown 610.
- Pending: A + head 1.25 (clean 5.32, aug 4.64, ext agree 0.83, KL 0.137).

- 22 Nov 01:00: Mirado and Philibert both gave a standing go: Philibert's Claude writes and directs the rounds; Mirado can overrule. Philibert may run steps directly (GPU queue, files in out/, append to logs/queue_v2.txt).

## Round 3 results / round 4 (22 Nov, 01:40)
- Best public **0.362** = --clean-frac 0.0 (caps 0.002-0.02 + --external, lr 2e-4 x 35 ep) with head scale 1.25.
  At head x1.25: clean-frac 0.5 / 0.2 / 0.0 = 0.304 / 0.3398 / 0.362; clean loss 5.25 / 4.64 / 3.80, aug ~4.0:
  clean images were over-forgotten.
- --other-tau 0.5: student cannot reach the sharper targets in 35 epochs (forget_kl 0.68 vs 0.2), forgetting collapses.
- Review (Philibert): score / (R x S) is ~constant over 8 runs (S = loss mean + std clean + aug, R = 1/(1+ext KL)):
  lr / epochs / alpha / soups only move along that line. Loss std is never targeted. 'aug' uses the training
  augmentations only.
- Round 4: 4a diag of the best model under 15 unseen alterations; 4b total head scale 1.40; 4c new options
  --cap-spacing log and --class-mix w (commit 5672238). Keep the raw checkpoint of every submitted model.

## Rounds 4-9 summary (22 Nov, 03:20)
Best public **0.3888** = out/r7b_capcount01_s1.4.pt: `unlearn_distill.py --conf-min 0.001 --conf-max 0.1 --cap-by count
--external external_images.npz --lr 2e-4 --epochs 35 --clean-frac 0.0` then `soup.py <ckpt> --head-scales 1.4`
(local: clean loss 3.43 (1.79), aug 3.70 (2.02), ext KL 0.082 before head scaling).
Head scale peak ~1.40 (1.25 / 1.40 / 1.55 = 0.362 / 0.3801 / 0.3471 uniform; 0.388 / 0.3839 at 1.40 / 1.55 graded).
Plateau 0.385-0.389: forgetting level and per-class gradation (--cap-by count) are each worth ~0.005.
Dropped (each compared at equal forgetting): --cap-spacing log (loss std unchanged), --class-mix 0.5 (flatter,
forgets less), --other-tau 0.75 (0.3621 at 1.40), --train-blocks 8 (ext KL 0.097), --train-blocks 2 (no retain gain),
mask45 (0.0, gate). Forgetting transfers to 15 unseen alterations (acc <= 0.11, only crop 60 % weaker at 0.18).
Infra: outputs reserved at start (inode quota), submitter skips files < 100 MB, data on GPU + 64 CPU cores:
35-epoch run ~4 min. Server cooldown 10 min.
Open: Caltech-256 retain anchor (Philibert, waiting for his go). Final file = best public (r7b_capcount01_s1.4.pt)
unless something beats it; which submission counts for the final ranking is still unconfirmed by the organizers.
