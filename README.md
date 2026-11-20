# hackaton_meta_20-11-2026

Our build for the **European AI Championship Paris**, the Paris stage of the European AI Championship led by CISPA Helmholtz Center for Information Security, co-organized by PoC Innovation with Meta and Epitech.

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
Every teammate's coding agent coordinates on one pinned GitHub issue, the **Message Board**. Rules for agents and humans: [`AGENTS.md`](AGENTS.md).

Setup, once per teammate:
```bash
gh auth login
git clone git@github.com:Landry-16/hackaton_meta_20-11-2026.git
cd hackaton_meta_20-11-2026
tools/board whoami
```
Claude Code picks the board up through the hooks in `.claude/`. Other agents (Codex, Cursor...) read `AGENTS.md` and run `tools/board unread` before each task.

Board commands:
```
tools/board post "[Name's Claude] text"   post (prefix required)
tools/board unread                        new posts since last check, flags mentions
tools/board wait                          block until someone posts
tools/board read -n 20                    last 20 posts
tools/board whoami                        check gh works
```

Board script and hooks come from https://github.com/azizu06/hackathon-agent-board.
