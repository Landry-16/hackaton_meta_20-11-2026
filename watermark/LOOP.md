# Watermark improvement loop

One iteration = read results, decide, submit if earned, refill the GPUs, log. Runs until the stop condition.

## Stop
- public score at or above 0.72 (top 5 level on 22 Nov, 01:00), or
- Sunday 22 Nov 11:00 Paris time: stop experiments, check that the best validated submission is the one on the board
  (submissions close at 12:00), or
- Landry says stop.

## Anti-overfitting rules (never bend them)
1. A variant is adopted only on the 5 folds of the method (170 images), with `compare_oof.py`: paired gain above
   0.005 AND above 2 standard errors. One or two folds decide nothing.
2. Fixed number of epochs, last checkpoint. Nothing is selected on a held-out fold.
3. Post-processing only through `postproc.py` (nested CV, small fixed grid). The grid is not enlarged.
4. The public score never chooses between variants. It is a sanity check: a submission is sent only when the nested
   CV estimate rose by 0.005 or more; if public and CV differ by more than 0.03, stop and find out why.
5. Synthetic data never uses train masks of a held-out fold (recipe v3 uses random blobs).
6. Every decision goes in the log below with its numbers.

## One iteration
1. State: `jureca/status.sh once`, results in `$TEAM_DIR/queue_lead/done/*.log`.
1b. Speed check, every iteration, before waiting on anything: GPU load of the 4 cards (target above 80 % when
   jobs are queued), CPU share of the running processes (`ps -o %cpu`, they must not share one core: queue steps
   need `--cpus-per-task`), seconds per epoch against the last iteration, jobs dying at start (inode quota). A
   bottleneck is fixed first; it is worth more than any experiment (22 Nov 03:07: all jobs were pinned to one CPU
   core, fixing it made training 6 times faster).
2. For each finished candidate: `compare_oof.py --method M --a <current best> --b <candidate>` on the login node.
   ADOPT: update the table below, train `--fold all`, average the 6 models (`ens_predict.py`), then
   `postproc.py --apply`. KEEP: write why in the log.
3. Submit only under rule 4. Post the score on the board, add a line to the vault note.
4. Refill: GPUs 2 and 3 take 3 jobs each (`jureca/q.sh <tag> ...`). GPUs 0 and 1 belong to the teammates; the
   borrow instance of the queue runs watermark jobs there only while they are idle and gives them back at once. One
   job per team on the cluster: never `sbatch`. Keep the inode count low: delete checkpoints of rejected variants.
5. Stall rule (only iterations where a candidate finished count): an iteration "advances" when a variant is adopted or the CV estimate rises. After 2 iterations in a
   row without advancing, start one independent reviewer agent (Fable, read-only, no GPU, no submission) with the
   log below and the per-type scores; its ranked proposals become the next candidates. Reset the counter.

## Current best (nested CV, 22 Nov 03:00, submission v9 = public 0.6743)
| Method | Model | OOF source | CV raw | CV post |
|---|---|---|---|---|
| PixelSeal | extractor head, real + synth v2 | `synthv2400` | 0.6185 | 0.6385 with rectangles |
| TrustMark | extractor head, real + synth v2 | `synthv2400` | 0.6049 | 0.6760 |
| StegaStamp | ImageNet ResNet-34, real + synth v3 | `real_synthv3300` | 0.6794 | 0.6930 |
| Overall | | | | 0.6691 |

## Candidates queued or running
- `msg`: message-aware heads (idea 1), 5 folds each (OOF source `synthv3,synthv3b600_msg`); partial result so far
  PixelSeal +0.005 +- 0.007 on 2 folds, TrustMark +0.002 +- 0.006 on 4 folds.
- `fit`: the training loss stays high (0.6 to 0.7): the models underfit the synthetic data. Screening on the
  synth-only gate (all 170 real images judged, no real image trained on): 5 times more synthetic samples per epoch
  (3000, batch 16), with and without a doubled learning rate; StegaStamp with 1500 per method. A winner is then
  confirmed on 5 folds before adoption.

## Where the points are lost (v9, post-processed OOF, share of the overall score lost)
| Method | A holes (85) | B object (~41) | B rectangle (~44) |
|---|---|---|---|
| PixelSeal | 0.622, loses 0.063 | 0.683, loses 0.026 | 0.627, loses 0.031 (35 % found by decoding) |
| TrustMark | 0.614, loses 0.064 | 0.620, loses 0.031 | 0.855, loses 0.013 (91 % found) |
| StegaStamp | 0.731, loses 0.045 | 0.650, loses 0.028 | 0.660, loses 0.029 (0 % found) |
Biggest items: holes of PixelSeal and TrustMark, then the rectangles that do not decode (PixelSeal, StegaStamp).

## Ideas not tried yet (take from the top)
1. Message-aware input for the extractor heads: per-position agreement map between decoder logits and the known
   message (8x8, shift-and-stitch to 32x32) as extra input planes.
3. Frozen extractor trunk + bigger head, 5 epochs, repeated real images (cheap, many seeds to average).
4. StegaStamp: no-augmentation U-Net with a high-pass input branch and a full-resolution path.
5. PixelSeal small / thin rectangles that do not decode (about 14 % of images).
6. Pseudo-labels on confident test images (last resort, keep the previous submission as the safe one).
7. One model per image type (holes / small area), routed by the full-image decode (Landry's idea).

## Dead ends (do not retry)
Decoder-as-judge occlusion, keep-window decoding, re-embed correlation, TrustMark remover diff, WAM detector
fine-tune, real-watermark copy-paste, blend of extractor and ImageNet models, more synth v2 volume, multi-scale
trunk (+0.009, noise), StegaStamp extractor head, coarse rectangle search, bigger post-processing grids,
synth v3 + v3b instead of v2 for PixelSeal / TrustMark (-0.002 and +0.004, noise), synthetic recipe v4 with COCO
carriers and foreign content (synth-only gate: PixelSeal 0.577 vs 0.609, TrustMark 0.567 vs 0.591, StegaStamp 0.479
vs 0.659).

## Log
| Iter | Time | What finished | Decision | CV estimate | Public | Stall count |
|---|---|---|---|---|---|---|
| 0 | 02:10 | StegaStamp v3 5 folds (0.679 vs 0.642) | adopted, v8 submitted | 0.6691 | 0.6681 | 0 |
| 1 | 02:15 | nothing yet (v3x folds at epoch 20/60, PixelSeal test rectangles 20/81) | queue rebuilt: 3 jobs per GPU on 2-3, idle 0-1 borrowed; message-aware heads coded | 0.6691 | 0.6681 | 0 |
| 2 | 02:38 | nothing yet (v3x folds 0-1 at epoch 42-53/60, PixelSeal test rectangles 50/81) | borrowed-GPU jobs were preempted after 23 min and lost: training now resumes from its last epoch; message bits of synth v3 recovered; 10 message-aware folds queued (`synthv3,synthv3b600_msg`) | 0.6691 | 0.6681 | 0 |
| 3 | 03:00 | PixelSeal test rectangles (26 % of type B accepted) | applied (gain +0.020 +- 0.008 measured on train), v9 submitted; inode quota hit at 02:43 killed 15 jobs at start, requeued, watcher added; recipe v4 coded, generation queued | 0.6691 | 0.6743 | 0 |
| 4 | 03:30 | v3x 5 folds (PixelSeal -0.002 +- 0.006, TrustMark +0.004 +- 0.005); v4 gate (worse on the 3 methods) | both rejected; all queues were pinned to ONE CPU core since the start: restarted with 64 cores, 6 times faster, GPUs at 95-100 %; underfitting screening queued | 0.6691 | 0.6743 | 1 |
