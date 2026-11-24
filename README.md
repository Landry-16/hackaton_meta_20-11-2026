# hackaton_meta_20-11-2026

Our build for the **European AI Championship Paris**, the Paris stage of the European AI Championship led by CISPA Helmholtz Center for Information Security, co-organized by PoC Innovation with Meta and Epitech.

> Setup (agent board and docs) prepared on 20 Nov, before the challenges were revealed. All project code is written during the hackathon.

## Event
| | |
|---|---|
| Where | Epitech, 24 Rue Pasteur, 94270 Le Kremlin-Bicêtre |
| Friday 20 Nov | 19:00 to 21:00, optional pizza and networking at KB2 |
| Saturday 21 Nov | 12:00 check-in and lunch, talks by Meta and CISPA, challenges revealed, **15:00 hacking starts** |
| Sunday 22 Nov | **12:00 hacking ends**, demos in front of the jury, awards, 15:00 end |
| Teams | 3 to 4 |
| Prizes | 10,000 EUR cash, offered by Meta |
| Bring | laptop, charger, ID |

Event page: https://luma.com/iwlbokns

## Reading list (shared by the organizers)
The three papers are about **trust in AI systems**: checking that a model forgot what it should, that an API serves the model it claims, and where a piece of content came from. Expect the challenges to be in this space (CISPA is a security lab).

1. **Towards Unbounded Machine Unlearning** (Kurmanji, Triantafillou et al., 2023), https://arxiv.org/abs/2302.09880. Removing a subset of training data from a trained network. Different goals (remove bias, fix mislabelled data, user privacy) need different "forgetting" metrics; privacy is measured with a membership inference attack. Proposes **SCRUB**, consistently strong on forget quality while keeping accuracy.
2. **Model Equality Testing: Which Model Is This API Serving?** (Gao, Liang, Guestrin, ICLR 2025), https://arxiv.org/abs/2410.20247. A two-sample test (Maximum Mean Discrepancy, simple string kernel) tells whether an API's outputs match a reference model, with about 10 samples per prompt. 11 of 31 commercial Llama endpoints did not match Meta's weights (quantized, finetuned, watermarked).
3. **Pixel Seal: Adversarial-only training for invisible image and video watermarking** (Souček, Fernandez et al., Meta, 2025), https://arxiv.org/abs/2512.16874. Invisible watermarks robust to crops, compression and re-encoding, trained adversarially only, adapted to high resolution and to video. Code and models: https://github.com/facebookresearch/videoseal.

Before Saturday: skim the three abstracts, and try `videoseal` locally if you have a GPU or a Colab.

## How we work

Use the pinned [general board](https://github.com/Landry-16/hackaton_meta_20-11-2026/issues/1) for decisions across tasks. Each challenge has its own issue for experiments and submissions. The [team rules](AGENTS.md#2-general-board-and-task-boards) describe mentions, handoffs and run announcements.

From the clone, with Python 3 and an authenticated GitHub CLI:

```bash
export BOARD_AGENT=Codex      # or Claude, or a distinct session name
tools/board follow model_equality
tools/board status            # current task header
tools/board unread            # task activity, direct mentions and general alerts
tools/board wait              # same routing, blocks until there is a match
```

For the next hackathon, create a general issue and one issue per challenge from the issue templates. Set their numbers in [the board directory](.github/boards.json) and link the task boards from the general issue. To use just one board, leave the tasks map empty. `BOARD_ISSUE` overrides the general issue, `BOARD_REPO` overrides the repository and `BOARD_TASK` overrides a saved subscription.

Board script and hooks adapted from https://github.com/azizu06/hackathon-agent-board.
