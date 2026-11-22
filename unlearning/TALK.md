# Task 3 (unlearning): 2-minute talk, Mirado + Philibert

Spoken script, about 130 words per speaker. Swap the parts if you prefer.

## Mirado (about 1 min): the problem and the key insight
"Our task: make a vision transformer forget 3,500 images, so that it behaves like models retrained without them.
We could not access those 'gold' models.

Our first idea was the textbook one: push the model to get the forgotten images wrong. That scored 0.14, because
forgetting too much is as visible as not forgetting: a model that is suddenly very wrong on specific images gives
them away.

So we changed the goal: we do not maximise forgetting, we imitate the right amount of ignorance. We distil the model
towards a target answer: a hesitant, plausible distribution, with an explicit level of surprise. That alone took us
from 0.14 to about 0.40."

## Philibert (about 1 min): structure, retain, and how we tuned it
"Then we found structure in the forget set: 9 classes are removed completely, 36 only partly. A model that never saw a
class is very surprised and confidently picks another class; for a partly removed class it only hesitates. Giving the
two groups their own loss level was our biggest single gain.

To keep everything else intact without the original training data, we anchored the model to the original on
generated images (text-to-image plus model inversion), and tuned the anchor weight: too strong, and it undoes the
forgetting.

We tested one change per submission, fitted a small model of the score from our own results (it predicted our
0.4762 exactly), and finished with a weight average of several seeds: 0.479 public, up from 0.14."

## Likely questions
- **Why not just maximise the error on the forget set?** The score compares us with retrained models; over-forgetting
  is detectable and scored lower (0.14 when we did it).
- **How did you keep accuracy without the training data?** Synthetic anchor images (text-to-image generation and
  model inversion), distilling the original's answers on them. Accuracy stays well above the 70 % gate.
- **How do you know you did not overfit the public 30 %?** One change per test; gaps under about 0.003 (our measured
  public noise) were never counted as wins; the final file is a seed average of a recipe that won on several tests.
- **What would you do with more time?** Match the gold models image by image, not only per class: our remaining loss
  is in which wrong class each image gets, and averaging teachers trained independently would target it.
- **Which part of the score was hardest?** Forgetting: retain is about 0.97, forgetting about 0.49.
