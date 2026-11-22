# Task 3 (unlearning): 2-minute talk on slides 6 and 7, Mirado + Philibert

Follows the team deck (teambuilders_solutions.pptx). About 1 minute per speaker; the same text is in the slide notes.

## Slide 6, Mirado (about 1 min): "Forget 3,500 images, keep everything else"
"Task three, unlearning. A vision transformer with 100 classes was trained on data that includes these 3,500 images.
We must return a model that behaves like the gold models, retrained without them. We never see the gold models, and
we have no retained data: the server is the only judge, about six submissions per hour.

The trap is to forget as hard as possible. A model that is suddenly very wrong on specific images gives them away:
forgetting too much scores as badly as not forgetting, and that is where we started, around 0.14.

So we do not maximise forgetting, we imitate the right amount of ignorance. This is teacher-student distillation on
the last four blocks. On each forget image, the student learns a target we build: the shape of the wrong answer, plus
a loss level we choose. On 30,000 images we generated ourselves, it must keep the original's predictions. At the end
we average the weights of several seeds.
Philibert shows what moved the score."

## Slide 7, Philibert (about 1 min): "0.27 to 0.483, one change per submission"
"Over the night the public score went from 0.27 to 0.483, one change per submission. Three ingredients.
One: two knobs per image. The wrong classes follow the original model's own ranking, sharpened, and the true-class
logit is set so that the loss hits the value we want.
Two: that value depends on the class. Nine classes lost all their roughly 130 images, so the gold model never saw
them: it is confidently wrong, loss near 10.7. The other 36 classes were only partly removed: the gold model is mildly
unsure, loss near 2.1. Giving each group its own level was our biggest jump.
Three: retain without retain data. A KL anchor on 30,000 generated images, then a weight average of four seeds. A
small formula fitted on our own scores predicted the retain gains to one thousandth.
What did not work: rebuilding the last blocks from scratch, and an anchor that is too strong, which undoes the
forgetting."

## Likely questions
- **Why not just maximise the error on the forget set?** The score compares us with retrained models; over-forgetting
  is detectable and scored lower (0.14 when we did it).
- **How did you keep accuracy without the training data?** Synthetic anchor images (text-to-image generation and
  model inversion), distilling the original's answers on them. Accuracy stays well above the 70 % gate.
- **How do you know you did not overfit the public 30 %?** One change per test; gaps under about 0.003 (our measured
  public noise) were never counted as wins; the final file is a seed average of a recipe that won on several tests.
- **What would you do with more time?** Match the gold models image by image, not only per class: our remaining loss
  is in which wrong class each image gets, and teachers trained independently would target it.
- **Which part of the score was hardest?** Forgetting: retain is about 0.97, forgetting about 0.49.
