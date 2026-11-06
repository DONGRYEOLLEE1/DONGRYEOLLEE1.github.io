---
layout: post
title: "Teaching a 2B Reasoner When to Stop Thinking"
subtitle: "Reward-only GRPO against Dr. GRPO, positional credit, think-only KL, GDPO, and a two-rollout Short-RL variant on one RTX 5080"
date: 2026-11-05 12:00:00 +0900
permalink: /research/thinkingcap-2b/
canonical_url: https://dongryeollee1.github.io/research/thinkingcap-2b/
lang: en
tags: [Reasoning, RL, GRPO, LoRA, Qwen, research]
categories: Research
use_math: false
comments: true
published: true
excerpt: "A controlled single-GPU study on Qwen3.5-2B: a correctness-gated length penalty alone cuts in-distribution thinking tokens by 75% and doubles strict accuracy, while Dr. GRPO normalization, positional credit, GDPO, and a Short-RL variant all fail to improve on it."
---

Small reasoning models do not just think too long. Often, they never stop. Under greedy decoding with a 4,096-token budget, [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) in thinking mode never emits `</think>` on **55%** of our in-distribution questions. We call this *non-termination*. On GSM8K, the gold answer already appears in the thinking of **83%** of those non-terminating responses.

ThinkingCap-2B, named after the BottleCap AI model that inspired it (Section 1), is our attempt to fix this with reinforcement learning on a single 16 GB consumer GPU, and to find out which refinements to GRPO, published or our own, actually help. We train six RL conditions, compare them with three untrained controls under one frozen and strict evaluation, and publish every per-item result and training curve.

## TL;DR {#tldr}

- **Non-termination, not verbosity, is the failure mode.** The base model never emits `</think>` within the budget on 55% of in-distribution questions (81% on GSM8K). On GSM8K, the gold number already appears in the thinking of 83% of those non-terminating responses (a string match, not proof that the model committed to it).
- **A correctness-gated length penalty in the reward is enough.** Plain GRPO with LoRA, two rollouts per question, and 200 updates (condition A) cuts mean thinking tokens by **75%** (2,848 → 703) and raises strict in-distribution accuracy from **42.75% to 85.25%**.
- **Loss-level refinements do not help.** Dr. GRPO normalization adds **+80%** thinking tokens and costs 3.4 pp; positional advantage decay adds another **+27%** and costs 6.4 pp; a think-only KL shows no measurable effect. The full bundle thinks **2.2×** as long as A.
- **GDPO and a two-rollout Short-RL variant do not beat A.** Both fail our predeclared efficiency rule: GDPO matches A's accuracy with **17%** more thinking, and ShortRL-G2 uses **32%** more. Both lose 5.5–7.5 pp on MATH-500.
- **A likely mechanism: at two rollouts, normalization is the length signal.** With two rollouts per question, GRPO's group-std normalization turns *any* reward gap, including a small length difference between two correct answers, into a full-strength advantage of about ±0.71. Dr. GRPO normalization, GDPO, and ShortRL-G2 each weaken that signal relative to correctness, and positional decay halves the credit on `</think>` itself. Replaying the logged rollouts supports this, but we did not vary the group size, and GDPO and ShortRL-G2 ran in a different training framework from A.

Three caveats travel with the numbers below: each condition is a single training seed; every evaluation uses greedy decoding with a 4,096-token budget, which makes loops more likely than Qwen's recommended sampling; and switching thinking off entirely still wins out of distribution (71.0% against 58.5% on our MATH-500 subset).

[![Accuracy versus thinking tokens for all nine conditions]({{ '/assets/research/thinkingcap-2b/figures/fig01_tradeoff.png' | relative_url }})]({{ '/assets/research/thinkingcap-2b/figures/fig01_tradeoff.png' | relative_url }})

*Figure 1. Strict accuracy against mean thinking tokens for all nine conditions under one frozen evaluation (greedy decoding, 4,096-token budget, no forced closure). (a) In distribution: GSM8K + ARC-Challenge. (b) Out of distribution: a MATH-500 subset. The arrow marks 200 RL updates of reward-only GRPO. The no-thinking baseline sits at zero thinking tokens, but its answers still average about 560 tokens in distribution and about 1,660 on MATH-500. Click any figure for the full-resolution image.*

**Contents**

1. [Small reasoners overthink, and often never stop](#h-motivation)
2. [Background](#h-background)
3. [Setup](#h-setup)
4. [Results at a glance](#h-results)
5. [Finding 1: RL first teaches the model to stop](#h-stop)
6. [Finding 2: No loss-level refinement helped, and two of three slowed compression](#h-ablation)
7. [Finding 3: GDPO and ShortRL-G2 do not beat the plain reward](#h-followups)
8. [A likely mechanism: with two rollouts, normalization is the length signal](#h-mechanism)
9. [Where the saved tokens come from](#h-analysis)
10. [Free baselines: brevity prompts backfire, no-thinking wins OOD](#h-controls)
11. [Limitations](#h-limitations)
12. [Conclusion](#h-conclusion)


## 1. Small reasoners overthink, and often never stop {#motivation}

Reasoning models learned to think at length because length paid off. In DeepSeek-R1-Zero's reinforcement learning, rule-based accuracy and format rewards alone raised AIME 2024 pass@1 from 15.6% to 77.9%, and the average response kept growing throughout training ([Guo et al., 2025](https://arxiv.org/abs/2501.12948)). A parallel literature has since measured how much of that thinking is wasted. On the question *"what is 2 plus 3?"*, o1-like models spend on average 1,953% more tokens than conventional models, and QwQ-32B-Preview writes 13 solutions over 901 tokens, although its first answer was already correct after 39 tokens ([Chen et al., 2024](https://arxiv.org/abs/2412.21187)). Accuracy typically follows an inverted U in chain-of-thought length: the optimal length grows with task difficulty but *shrinks* as models get more capable ([Wu et al., 2025b](https://arxiv.org/abs/2502.07266)). Surveys now organize dozens of methods for shortening reasoning ([Sui et al., 2025](https://arxiv.org/abs/2503.16419); [Feng et al., 2025](https://arxiv.org/abs/2504.10903)).

Small models add a harsher failure mode: **the thinking never ends**. At temperature 0, DeepSeek-R1-Distill-Qwen-1.5B loops in 76% of its AIME responses, against 37% for the 32B model distilled the same way ([Pipis et al., 2025](https://arxiv.org/abs/2512.12895)). Qwen's own model card warns that in thinking mode Qwen3.5-2B *"is more prone to entering thinking loops compared to other Qwen3.5 models, which may prevent it from terminating generation properly"*, and ships the model with thinking switched off by default ([Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B)). An independent benchmark of thinking/non-thinking hybrid models points the same way at 2B, although it does not measure loops. Under greedy decoding with a 32K budget, Qwen3.5-2B averages 11.8% in full-thinking mode and 18.7% with thinking off over five reasoning benchmarks, while using about half as many tokens in the latter mode ([Ning et al., 2026](https://arxiv.org/abs/2605.28398)). The cost is real, too. In the AI Energy Score measurements, three models tested with reasoning both on and off (DeepSeek-R1-Distill-Llama-70B, Phi-4-reasoning-plus, and SmolLM3-3B) used 150–700× more energy per query with reasoning on, mostly because they generated far more tokens ([Luccioni & Gamazaychikov, 2025](https://huggingface.co/blog/sasha/ai-energy-score-v2)).

The direct inspiration for this project is BottleCap AI's [ThinkingCap-Qwen3.6-27B](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B) (July 2026). It reports 45.8% fewer thinking tokens over 12 out-of-domain benchmarks at a 0.8-point accuracy cost, 57.7% fewer on in-domain splits with a small accuracy gain, and, out of domain, a drop in traces truncated before closing their reasoning from 2.9% to 0.4% (1.6% to 0.03% in domain). The training recipe is disclosed only in outline: the card describes fine-tuning *"with state-of-the-art algorithms"* on a curated set of problems, and the [accompanying blog post](https://bottlecapai.com/thinkingcap-qwen3-6-27b) says the objective *"rewarded efficient reasoning rather than simply rewarding correctness."* We wanted to know whether a fully specified, open recipe reproduces that effect at the other end of the scale, and which refinements, published or new, are worth their complexity there. We asked three questions:

1. **Reproduction.** Does a correctness-gated length penalty work at 2B parameters, with LoRA, two rollouts per question, and one 16 GB GPU?
2. **Attribution.** Loss-level refinements could plausibly sharpen compression: Dr. GRPO's unbiased loss normalization from the literature, plus two components we designed ourselves, positional credit inside the reasoning span and a KL restricted to that span. Which of them earns its keep?
3. **Follow-ups.** Do newer algorithms built for multi-objective or length-aware rewards, GDPO and Short-RL, improve on the plain reward in this regime?

## 2. Background {#background}

### 2.1 GRPO, and RL on a small budget {#bg-grpo}

GRPO ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)) removes PPO's learned critic ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)). It samples a group of $$G$$ outputs for each prompt, gives every token of output $$o_i$$ the same group-standardized advantage, averages the clipped token losses within each response and then across the group, and adds a KL penalty toward a frozen reference:

$$
\hat{A}_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}
$$

$$
\mathcal{J}(\theta) = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\!\Big(\rho_{i,t}\hat{A}_i,\ \operatorname{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)\,\hat{A}_i\Big) - \beta\,\mathbb{D}_{\mathrm{KL}}\!\left[\pi_\theta \,\|\, \pi_{\mathrm{ref}}\right]
$$

Reasoning RL normally uses 8 to 64 rollouts per prompt, which does not fit a 16 GB card alongside training. Two rollouts can be enough. 2-GRPO keeps 97.6% of 16-rollout GRPO's performance with 12.5% of the rollouts ([Wu et al., 2025a](https://arxiv.org/abs/2510.00977)), although that analysis assumes binary rewards. A length-shaped reward, as Section 8 shows, behaves differently. LoRA is not the bottleneck either. A policy-gradient episode carries roughly one bit of information, and LoRA matches full fine-tuning for RL *"even with ranks as low as 1"* ([Schulman et al., 2025](https://thinkingmachines.ai/blog/lora/)). Tina trains a 1.5B reasoner with LoRA-GRPO for about $9 ([Wang et al., 2025b](https://arxiv.org/abs/2504.15777)). Small-budget studies also report the failure we target. Under a 4,096-token cap, a GRPO-trained 1.5B model was *"often exceeding the 4096-token limit before producing a final answer"* ([Dang & Ngo, 2025](https://arxiv.org/abs/2503.16219)).

### 2.2 Length-aware rewards {#bg-length}

Most length-control methods add a length term to the reward, and they differ mainly in *when* the term applies:

| Method | Length signal | Gated by correctness? | Reported effect |
|---|---|---|---|
| Kimi k1.5 ([Kimi Team, 2025](https://arxiv.org/abs/2501.12599)) | Min–max-normalized length within the group, switched on after a warm-up phase | Asymmetric: correct answers can gain, wrong ones only lose | Long-to-short model: 60.8 AIME 2024 pass@1 at 3,272 tokens |
| Arora & Zanette ([2025](https://arxiv.org/abs/2502.04463)) | 1[correct] × (1 − α·σ(length standardized over correct rollouts)) | Yes, multiplicative | About 50% fewer tokens at under 5% accuracy loss (7B) |
| L1 / LCPO ([Aggarwal & Welleck, 2025](https://arxiv.org/abs/2503.04697)) | Distance to a token budget written in the prompt | L1-Max yes, L1-Exact no | Follows the prompted budget |
| ThinkPrune ([Hou et al., 2025](https://arxiv.org/abs/2504.01296)) | Reward only if the answer fits under a shrinking cap (4K → 3K → 2K) | Implicit, through truncation | About half the AIME 2024 length for about 2 points (1.5B) |
| DLER ([S.-Y. Liu et al., 2025](https://arxiv.org/abs/2510.15110)) | Zero reward beyond 4,000 tokens, plus batch-wise reward normalization, a higher clip, and dynamic sampling | Implicit, through truncation | 77% (1.5B) and 69% (7B) fewer tokens, with higher AIME 2024 accuracy |
| Short-RL ([Yuan et al., 2025](https://arxiv.org/abs/2505.12284)) | Kimi-style term behind three gates (Section 2.4) | Yes, plus an accuracy switch | 40% (logic) and 11–33% (math) shorter training-time responses |
| **ThinkingCap-2B (ours)** | 1[closed and correct] × (1 − 0.4·min(L/768, 1)) | Yes, multiplicative, plus closure | Section 4 |

Our reward is closest to Arora & Zanette's multiplicative gate, with a fixed normalizer instead of group statistics, which are meaningless with two samples. The requirement that `</think>` close within the rollout budget plays the role of a truncation term. DLER makes a point that turns out to be central here: simple length penalties tend to lose accuracy because of how they are *optimized*, not because of their form, and batch-level reward normalization is one of the levers it changes.

### 2.3 Loss-level refinements {#bg-loss}

Several papers argue that GRPO's normalizers bias response length. Dr. GRPO identifies two biases ([Z. Liu et al., 2025](https://arxiv.org/abs/2503.20783)). The per-response $$1/\lvert o_i\rvert$$ factor gives short correct answers larger per-token updates and lets long wrong answers off lightly, so wrong answers drift longer. Dividing by the group standard deviation up-weights prompts whose rewards barely vary. Dr. GRPO removes both, using $$\hat{A}_i = r_i - \operatorname{mean}(r)$$ and a constant loss denominator. DAPO normalizes at the token level instead, raises the upper clip, filters groups whose rewards are all equal, masks overlong samples, and drops the KL term ([Yu et al., 2025](https://arxiv.org/abs/2503.14476)).

Other work asks *where* in a response the credit should go. About 20% of chain-of-thought tokens have high entropy. These "forking" tokens, such as *wait*, *however*, and *thus*, carry most of what RL changes, and restricting the gradient to them matches or beats full updates ([Wang et al., 2025a](https://arxiv.org/abs/2506.01939)). The KL term is also contested. DeepSeekMath and R1 keep a k3 penalty in the loss, while DAPO and Dr. GRPO drop it, and estimator studies find that only some placements give an unbiased gradient ([Shah et al., 2025](https://arxiv.org/abs/2512.21852)). None of these works evaluates a KL restricted to the thinking span, or a positional decay of advantages within it. Our components D and K test exactly those two ideas.

### 2.4 Multi-reward normalization and lazy length penalties {#bg-followups}

**GDPO** ([S.-Y. Liu et al., 2026](https://arxiv.org/abs/2601.05242)) starts from a normalization collapse. If several rewards are summed before GRPO's group normalization, distinct reward combinations can map to the same advantage. Their own example is two rollouts with two binary rewards: totals of (0, 1), (0, 2), and (1, 2) all become (−0.71, +0.71). GDPO instead normalizes each reward within the group, sums the results, and re-normalizes across the batch. With 16 rollouts and a ≤4,000-token length reward on DeepSeek-R1-1.5B, it reaches 29.4% AIME accuracy and a 6.5% over-length rate, against 23.1% and 10.8% for GRPO (the untrained model scores 29.8%, with 91.5% of its responses over length).

**Short-RL** ([Yuan et al., 2025](https://arxiv.org/abs/2505.12284), now titled *"Shorten After You're Right: Lazy Length Penalties for Reasoning RL"*) finds that applying a Kimi-style length reward from the first step collapses outputs, and adds three gates. *RightGate* applies the term only to correct responses. *SlackBand* treats responses within a tolerance of the shortest correct one as equal. *StableSwitch* turns the term on only while batch accuracy is within a margin of its running maximum. With 8 rollouts it reports 40% shorter training-time responses on logic puzzles and 11–33% shorter on math, with accuracy maintained or improved.

Both methods were tuned with 8–16 rollouts per prompt and large batches. With two rollouts, per-prompt minima, maxima, and per-reward group statistics become nearly degenerate. Our follow-up runs therefore test **two-rollout adaptations** of these ideas, not the original methods.

### 2.5 Zero-cost alternatives {#bg-zero}

Three cheaper routes deserve a fair comparison. **Skipping the thinking**: prefilling a closed thinking block ("NoThinking") beats regular thinking at matched token budgets on many benchmarks, for example 51.3 against 28.9 on AMC 2023 at 700 tokens ([Ma et al., 2025](https://arxiv.org/abs/2504.09858)). Hybrid models such as Qwen3 build the switch in during training ([Yang et al., 2025](https://arxiv.org/abs/2505.09388)). **Capping the thinking at decode time**: budget forcing appends the end-of-thinking delimiter once a budget is reached ([Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393)), and Qwen3's thinking budget does the same in production. **Asking for brevity**: concise-CoT prompts shorten large API models ([Renze & Guven, 2024](https://arxiv.org/abs/2401.05618); [Xu et al., 2025](https://arxiv.org/abs/2502.18600)), but Chain of Draft degrades sharply at 1.5–3B scale. Stated token budgets that are too tight also make models produce more tokens than a moderate budget does (157 against 86 in TALE's example, though still fewer than the 258 of unconstrained CoT), which TALE calls *token elasticity* ([Han et al., 2024](https://arxiv.org/abs/2412.18547)). We include the first and third routes as controls and discuss the second in Section 11.

## 3. Setup {#setup}

### 3.1 Model, data, and training {#setup-training}

**Model.** [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) has a hybrid-attention architecture: 24 layers in six blocks of three Gated DeltaNet (linear-attention) layers followed by one gated full-attention layer. It runs in non-thinking mode by default, so we switch thinking on through the chat template. Every RL condition trains a fresh LoRA adapter ([Hu et al., 2022](https://arxiv.org/abs/2106.09685); rank 16, alpha 32, no dropout) from the same base weights.

One practical detail is easy to miss. The LoRA target list that is standard for Llama-style models (`q,k,v,o,gate,up,down`) **silently skips the attention projections of all 18 linear-attention layers** of this architecture; only their MLPs get adapters. We add their main input and output projections (`in_proj_qkv`, `out_proj`) explicitly; their three other projections (`in_proj_z`, `in_proj_a`, `in_proj_b`) get no adapters.

**Data.** The training pool holds 10,717 verifiable questions from the GSM8K and ARC training splits. In 200 updates × 8 questions, each run consumes the same frozen 1,600 questions (1,124 GSM8K and 476 ARC), each once and in the same order. The 300-question dev set is drawn from pool items that no run ever trains on.

**Optimization.** We use two responses per question (G = 2), so each update sees 16 rollouts. The remaining settings are shared by every RL condition:

| Setting | Value |
|---|---|
| Updates | 200 (seed 0) |
| Learning rate | 5e-6, constant after 10 warmup updates |
| KL coefficient (k3 estimator) | 0.02 |
| PPO clip | 0.2 |
| Rollout sampling | temperature 1.0, top-p 0.95, top-k 20 |
| Rollout budget | 1,280 new tokens |
| Truncated rollouts | kept in the group statistics, but masked out of the loss |
| Hardware | one RTX 5080 (16 GB), WSL2 |

The original four conditions use [TRL](https://github.com/huggingface/trl) 1.9.0. GDPO and ShortRL-G2 use native verl 0.5.0 ([Sheng et al., 2024](https://arxiv.org/abs/2409.19256)) with the same hyperparameters, data order, and evaluation. Section 11 explains why this framework change matters.

### 3.2 The reward {#setup-reward}

BottleCap describes its objective only in outline. Our reward is a minimal, fully specified stand-in for it: a correctness gate times a capped length discount.

$$
r(o) \;=\; \mathbb{1}\big[\text{closed} \,\wedge\, \text{correct}\big]\cdot\Big(1 - \alpha \min\big(L_{\text{think}}/B,\ 1\big)\Big), \qquad \alpha = 0.4,\; B = 768
$$

Here *closed* means the response emits `</think>` on its own, and *correct* is judged on the text after it. A correct answer therefore earns between 0.6 and 1.0. A wrong answer earns 0, and so does a rollout that never closes its reasoning span. The closure requirement matters most. It is the only thing that makes *finishing* the thought learnable, and no generation-time intervention such as forced closure or budget forcing is used anywhere. Because of the cap, two correct answers that both think for 768 tokens or more receive identical rewards.

### 3.3 Nine conditions {#setup-conditions}

| Condition | Stack | What changes relative to A |
|---|---|---|
| Base, thinking | — | Untrained; thinking enabled |
| Base, no thinking | — | Untrained; thinking disabled by the chat template |
| Base, "be brief" | — | Untrained; *"Answer using as few reasoning tokens as possible."* added to the prompt |
| **A (reward-only)** | TRL | Stock GRPO: group-std scaling and per-response length normalization |
| +R | TRL | Dr. GRPO normalization: no std scaling and a constant loss denominator |
| +R+D | TRL | + positional decay of positive advantages over the thinking span, from 1.0 down to 0.5 |
| B (+R+D+K) | TRL | + KL applied only to thinking tokens |
| GDPO | verl | Correctness and length components normalized separately per group, summed 0.6 / 0.4, then re-standardized across the batch |
| ShortRL-G2 | verl | Length penalty relative to the shorter correct sibling, with a 100-token slack, active only while rolling accuracy stays near its best |

The R → D → K order is deliberate. R is the borrowed, published technique, so adding it first asks *how much of the bundle is just Dr. GRPO* before the two novel components are judged on top of it.

### 3.4 One frozen, strict evaluation {#setup-eval}

Every evaluation result in this post comes from one frozen evaluation that all nine conditions share (the exceptions are training logs and results we mark as coming from our first screening):

- **Benchmarks.** Dev 300 (diagnostic), GSM8K test 400 ([Cobbe et al., 2021](https://arxiv.org/abs/2110.14168)) and ARC-Challenge test 400 ([Clark et al., 2018](https://arxiv.org/abs/1803.05457)), together *in-distribution* (**ID**), and a 200-question subset of MATH-500 ([Hendrycks et al., 2021](https://arxiv.org/abs/2103.03874); [Lightman et al., 2023](https://arxiv.org/abs/2305.20050)) stratified by difficulty level (*out-of-distribution*, **OOD**). Each test subset keeps the 24 items from our first screening (the first 24 rows of its split) and fills the rest with a seed-42 draw. They are not full official benchmark runs.
- **Decoding.** Greedy, at most 4,096 new tokens, bf16, and **no forced closure** of the reasoning span.
- **Strict accuracy.** A thinking response is correct only if it emits `</think>` naturally and the text *after* it contains the right answer: `math-verify` for math, the choice letter for ARC.
- **Tokens.** Thinking tokens run up to `</think>`, so an unclosed response counts all 4,096. Total tokens are everything generated, including `</think>` and EOS; prompts are excluded.
- **Statistics.** Prompt-paired bootstrap with 10,000 resamples, stratified by benchmark. ID metrics weight GSM8K and ARC equally.

That makes 9 × 1,300 = **11,700 saved responses**. We re-derived every token boundary and answer region from the raw token IDs, recomputed every table cell from per-item records, and reproduced every confidence interval independently.

> **A lesson about lenient scoring.** Our first screening pass used only 24 items per test benchmark. It also had a quiet bug: when the reasoning never closed, the harness still searched the *unfinished thinking text* for an answer. On the dev set, 87 of the base model's 207 "correct" answers came from responses that never finished thinking. That inflated its accuracy from 40.0% to 69.0%. The ranking of the trained conditions survived, but we re-ran every condition under the strict protocol above. If you measure token efficiency, make sure an answer the model never committed to cannot earn credit.

## 4. Results at a glance {#results}

| Condition | Dev | GSM8K | ARC-C | MATH-500 | ID acc. | ID thinking tok. | ID total tok. | ID closure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base, thinking | 36.67 | 18.00 | 67.50 | 5.00 | 42.75 | 2,848 | 2,974 | 44.9% |
| Base, no thinking | 88.33 | 77.00 | 88.00 | **71.00** | 82.50 | 0 | **562** | n/a |
| Base, "be brief" | 12.67 | 9.75 | 12.00 | 5.50 | 10.88 | 3,789 | 3,797 | 11.5% |
| **A, reward-only** | **90.67** | 77.25 | **93.25** | 58.50 | 85.25 | **703** | 916 | **93.9%** |
| +R | 80.33 | 73.75 | 90.00 | 41.50 | 81.88 | 1,265 | 1,503 | 88.0% |
| +R+D | 75.33 | 61.75 | 89.25 | 35.50 | 75.50 | 1,602 | 1,820 | 81.0% |
| B (+R+D+K) | 77.00 | 65.75 | 89.75 | 39.50 | 77.75 | 1,542 | 1,750 | 82.6% |
| GDPO | 90.33 | **79.50** | 91.75 | 53.00 | **85.63** | 825 | 1,060 | **93.9%** |
| ShortRL-G2 | 87.67 | 76.75 | 92.25 | 51.00 | 84.50 | 926 | 1,154 | 92.1% |

*Strict accuracy in %, token means per question over all ID responses, including incorrect and unclosed ones. Bold marks the best value in each column; for thinking tokens, the lowest among the thinking policies. The no-thinking base has no reasoning span, so it has no closure rate.*

[![Per-benchmark accuracy and thinking tokens]({{ '/assets/research/thinkingcap-2b/figures/fig02_benchmarks.png' | relative_url }})]({{ '/assets/research/thinkingcap-2b/figures/fig02_benchmarks.png' | relative_url }})

*Figure 2. Accuracy (top row) and mean thinking tokens (bottom row) on each benchmark. Each top panel labels the best accuracy of any condition, and each bottom panel labels the lowest thinking cost among thinking policies. A is the most frugal thinker on all four sets.*

Three patterns run through the table. Reward-only A, GDPO, and ShortRL-G2 form a tight cluster at about 85% ID accuracy, but they are separated by up to 32% in thinking cost. The TRL ladder gets worse with the first two loss terms, and the third changes little. And the untrained model with thinking *switched off* beats every trained thinking policy on MATH-500. Section 5 first shows how training gets off the ground; Sections 6–8 take the first two patterns, and Section 10 returns to the third.

## 5. Finding 1: RL first teaches the model to stop {#stop}

[![Training dynamics of the four TRL runs from W&B]({{ '/assets/research/thinkingcap-2b/figures/fig03_training_trl.png' | relative_url }})]({{ '/assets/research/thinkingcap-2b/figures/fig03_training_trl.png' | relative_url }})

*Figure 3. Training curves of the four TRL runs, drawn from the public W&B histories (thin: per update; thick: exponential moving average). (a) Mean rollout length. (b) Share of rollouts cut off at the 1,280-token training budget. (c) Mean reward. (d) Share of two-response groups whose rewards are identical, which therefore carry no reward signal.*

Training starts from almost nothing. At the first update, **94%** of rollouts hit the 1,280-token cap without finishing, and in the first 50 updates about **80%** of the two-response groups receive identical rewards (usually two zeros). With G = 2, such a group carries no reward signal. Most of these pairs are two truncated rollouts, which are masked out of the loss entirely; on the rest, only the small KL term acts. The signal bootstraps from the rare pair in which one rollout closes its reasoning and answers correctly, and the other does not.

Then the model learns to stop. Between updates 110 and 130 the reward climbs and truncation falls in every run, but only A keeps compressing. Over its last 10 updates, A truncates **16.9%** of rollouts and averages a reward of **0.63**. The loss-modified runs stall at 42–52% truncation and 0.30–0.38 reward.

[![Non-termination rates by condition and benchmark]({{ '/assets/research/thinkingcap-2b/figures/fig04_termination.png' | relative_url }})]({{ '/assets/research/thinkingcap-2b/figures/fig04_termination.png' | relative_url }})

*Figure 4. Share of responses that never emit `</think>` within the 4,096-token evaluation budget. A sharply reduces non-termination on every benchmark.*

At evaluation the same story appears. The base model's ID non-termination rate is **55.1%**; on GSM8K alone it is 81.0%. After 200 updates, A's ID non-termination falls to **6.1%**. On the paired ID set, A versus Base is **+42.50 pp** accuracy [+39.25, +45.75] with **−75.3%** thinking tokens [−77.5, −73.0]. Even on MATH-500, which no run trained on, A's non-termination drops from 94.5% to 39.5%, and its accuracy rises from 5.0% to 58.5%.

On GSM8K at least, much of the gain looks like termination rather than new reasoning: the gold number already appears in most of the base model's non-terminating responses (Section 9). The answers finally exist.

## 6. Finding 2: No loss-level refinement helped, and two of three slowed compression {#ablation}

[![Ablation forest plot]({{ '/assets/research/thinkingcap-2b/figures/fig05_ablation.png' | relative_url }})]({{ '/assets/research/thinkingcap-2b/figures/fig05_ablation.png' | relative_url }})

*Figure 5. The effect of each added component under the frozen protocol: (a) change in mean ID thinking tokens and (b) change in ID accuracy, with paired 95% bootstrap intervals (n = 800). Reference lines show the thresholds we fixed before the original screening: a component "helps" only if it shortens thinking by at least 10% without losing more than 5 pp of accuracy.*

The ladder gives a clean attribution:

- **R, Dr. GRPO normalization: harmful.** Adding R to A raises thinking tokens by **+80.1%** [+67.1, +94.6] and changes ID accuracy by **−3.4 pp** [−5.4, −1.4] and MATH-500 accuracy by −17.0 pp. A fix designed to remove GRPO's length and difficulty biases produces the largest jump in thinking length, and the largest MATH-500 loss, of any rung.
- **D, positional advantage decay: harmful.** It adds another **+26.6%** [+20.2, +33.5] thinking tokens and changes ID accuracy by **−6.4 pp** [−8.8, −4.0].
- **K, think-only KL: neutral.** Thinking changes by −3.8% [−8.3, +1.1] and accuracy by +2.2 pp [−0.1, +4.6]. Neither interval excludes zero, so we detect no effect of K in either direction (the accuracy interval only just includes zero).

The complete bundle B therefore thinks **2.2×** as long as A (+119.4% [+102.6, +138.3]) and is 7.5 pp less accurate in distribution. These verdicts match our first screening pass (the 300-question dev set plus 24 items per test benchmark): R harmful, D harmful, K neutral. The frozen sets do correct one claim in our [GitHub README](https://github.com/DONGRYEOLLEE1/reducing-think-token), a strict A > +R > +R+D > B ordering. In distribution, B is slightly *shorter and more accurate* than +R+D, which is consistent with K being neutral, not with a strict ladder.

## 7. Finding 3: GDPO and ShortRL-G2 do not beat the plain reward {#followups}

We then tested two newer recipes that target exactly this kind of multi-objective, length-aware reward. Both reach A's accuracy range. Neither reaches its efficiency.

[![Training dynamics of the verl runs from W&B]({{ '/assets/research/thinkingcap-2b/figures/fig06_training_verl.png' | relative_url }})]({{ '/assets/research/thinkingcap-2b/figures/fig06_training_verl.png' | relative_url }})

*Figure 6. Training curves of GDPO and ShortRL-G2 from W&B: (a) thinking tokens per rollout, (b) share of rollouts that close `</think>`, (c) rollout accuracy. These verl metrics are not directly comparable with the TRL metrics in Figure 3.*

Both runs follow the same arc as A. They start near 1,200 thinking tokens with about 18% closure and 12–13% rollout accuracy. Over the last 10 updates they reach 622 tokens / 81.9% closure / 75.0% accuracy (GDPO) and 676 / 74.4% / 68.8% (ShortRL-G2). GDPO's closure and accuracy are still rising at update 200. For ShortRL-G2, both peak around update 180, the last update with its length gate on, and dip over the final 20.

Before training, we declared one decision rule against A. A follow-up improves on A only if its point estimates satisfy all five conditions: ID accuracy loss of at most 2 pp, a loss of at most 3 pp on each ID benchmark, at least 10% fewer thinking tokens, no increase in total tokens, and at most a 2 pp increase in non-termination.

[![Predeclared decision rule]({{ '/assets/research/thinkingcap-2b/figures/fig07_decision_rule.png' | relative_url }})]({{ '/assets/research/thinkingcap-2b/figures/fig07_decision_rule.png' | relative_url }})

*Figure 7. Each follow-up against A under the predeclared rule. Squares are point estimates, bars are paired 95% bootstrap intervals, and the shaded side of the green line passes. Both methods fail both token criteria.*

| vs. A | ID acc. | Thinking tokens | Total tokens | MATH-500 |
|---|---:|---:|---:|---:|
| GDPO | +0.4 pp [−1.5, +2.2] | **+17.4%** [+8.4, +27.1] | +15.7% | **−5.5 pp** [−10.0, −1.0] |
| ShortRL-G2 | −0.8 pp [−2.6, +1.1] | **+31.8%** [+22.0, +42.4] | +26.0% | **−7.5 pp** [−13.0, −2.0] |

**GDPO** has the highest ID accuracy point estimate in the study, 85.63%. That lead over A is only 3 of 800 questions, and its interval spans zero. It pays for that lead with 17% more thinking and a significant **−5.5 pp** on MATH-500. Fourteen of the 16 MATH questions it loses relative to A are its own non-terminations. Its accuracy and closure passes are statistically supported, with the ARC per-benchmark gate as the one exception, but it fails the efficiency gates outright.

**ShortRL-G2** fails the same two gates by a wider margin. Roughly half of its extra thinking over GDPO comes from additional non-terminations, and half from longer reasoning on questions both models finish. Its MATH-500 deficit against A, −7.5 pp, is mostly additional non-termination: 90 of 200 responses never close, against 79 for A.

## 8. A likely mechanism: with two rollouts, normalization is the length signal {#mechanism}

Why would a borrowed normalization, a credit-shaping trick, and two newer algorithms all lose to the plainest recipe? We think most of the answer lies in what GRPO does with a group of two.

With G = 2 and the Bessel-corrected standard deviation that TRL uses, the standardized advantage of a pair collapses to a constant:

$$
\hat{A}_{1,2} \;=\; \pm\,\frac{\lvert r_1 - r_2\rvert/2}{\lvert r_1 - r_2\rvert/\sqrt{2} + \epsilon} \;\approx\; \pm 0.71 \qquad \text{for any } r_1 \neq r_2 .
$$

The size of the reward gap disappears; only its *sign* survives. So in A, the preference "the shorter of two correct answers is better" receives the **same full-strength update** as "a correct answer beats a wrong one", even when the two correct answers differ by only a few dozen tokens. That appears to be the length signal we needed, and each of the four changes below weakens it:

[![Reward shapes and the learning signal per group]({{ '/assets/research/thinkingcap-2b/figures/fig08_mechanism.png' | relative_url }})]({{ '/assets/research/thinkingcap-2b/figures/fig08_mechanism.png' | relative_url }})

*Figure 8. (a) A's reward and ShortRL-G2's reward as functions of thinking length. ShortRL-G2 is shown for one case: the gate is on and the shorter correct sibling used 300 tokens. (b) The size of the advantage that the preferred response receives in three typical two-response groups. GDPO values are shown before its batch-level re-standardization, which rescales all advantages in a batch by a common factor.*

- **R** removes the std scaling, so the advantage becomes the raw half-gap. A 100-token difference between two correct answers is now worth ±0.026, while a correct-versus-wrong pair is still worth ±0.3–0.5. Measured against correctness, the length preference becomes about **12–19× weaker** than in A, where the two are equal. R's constant loss denominator also removes GRPO's extra per-token weight on short sequences.
- **D** scales positive credit down to 0.5 at the end of the thinking span, and the `</think>` token itself sits at that end. It therefore halves the reinforcement of the very action that Section 5 shows RL learns first: **stopping**.
- **GDPO** z-scores correctness and length separately. With two responses, each z-score again collapses to ±0.71. After the 0.6 / 0.4 weighting, a length-only pair gets **0.28** against 0.71 for a correctness pair (0.42 if the correct answer thinks for 768 tokens or more), while A gives 0.71 to both. After batch re-standardization, GDPO's length advantages are not small in absolute size. They are small *relative to correctness*.
- **ShortRL-G2** adds two filters on top of the length penalty. Pairs within 100 tokens of each other are ignored, and the penalty applies only while 80-response rolling accuracy stays within 0.05 of its running maximum. One lucky window at update 180 set that maximum to 0.84, and the gate stayed off for the rest of training. Overall the gate was on in only **80 of 190** post-burn-in updates, and only **148 of 3,200** training rollouts were ever penalized. Of the 513 pairs in which both answers were correct, only 148 received a length-ordering advantage. A's reward would have ordered 487 of them.

This is a hypothesis about mechanism, supported by replaying the logged rollouts. It is not a controlled test, because we did not vary the group size. It does, however, account for the R, GDPO, and ShortRL-G2 results with one fact: **at G = 2, group-std normalization is not a nuisance that biases length; it is what makes a small length penalty bite.** In larger groups the standard deviation is a meaningful estimate of reward spread. At G = 2 it is just $$\lvert r_1 - r_2\rvert/\sqrt{2}$$, and dividing by it erases the size of the gap. Removing that division (R) or diluting its effect (GDPO, ShortRL-G2) takes the amplifier away.

The amplifier cuts both ways. Whenever both answers are correct, the longer one receives a full-strength *negative* advantage, the failure mode that DRPO describes for correctness-gated length rewards in GRPO ([Li et al., 2025](https://arxiv.org/abs/2510.04474)). On harder, out-of-distribution problems, where long reasoning is sometimes necessary, that pressure may be part of why A still trails the no-thinking baseline.

## 9. Where the saved tokens come from {#analysis}

[![Distribution of thinking length]({{ '/assets/research/thinkingcap-2b/figures/fig09_length_cdf.png' | relative_url }})]({{ '/assets/research/thinkingcap-2b/figures/fig09_length_cdf.png' | relative_url }})

*Figure 9. Cumulative distribution of thinking tokens per response. The vertical jump at 4,096 is the share of responses that never close. (a) ID, n = 800; (b) MATH-500, n = 200.*

Mean token counts mix two things: how long a model reasons when it closes its thinking, and how often it does not (every non-terminating response counts as exactly 4,096). The distributions separate them. A's median ID response thinks for **344 tokens on GSM8K and 351 on ARC**, while the base model's GSM8K median is the full budget. A also leads the follow-ups on items that both models close and on medians, so its advantage is not just fewer non-terminations.

[![Anatomy of overthinking]({{ '/assets/research/thinkingcap-2b/figures/fig10_overthinking.png' | relative_url }})]({{ '/assets/research/thinkingcap-2b/figures/fig10_overthinking.png' | relative_url }})

*Figure 10. (a) Occurrences of "wait" in the thinking span per ID response. (b) Self-check phrases ("wait", "hmm", "double-check", "let me verify", "alternatively", …) per 1,000 thinking tokens. (c) Among the 324 base GSM8K responses that never close, the share whose thinking already contains the gold answer within the first k tokens.*

The base model's long tail is full of self-checking and repetition. It writes "wait" **10.6 times** per ID response, and the "be brief" prompt makes this worse, at **35.6** times. After RL, A writes "wait" 0.9 times per response. The density of self-check phrases also falls, from 4.2 to 1.8 per thousand thinking tokens, so A is not simply the same reasoning shortened.

Panel (c) supports the "found it, then kept checking" story with numbers. Among the 324 base GSM8K responses that never close, the gold answer already appears within the first 512 thinking tokens in 46%, and within the first 1,024 tokens in 74% (a string match on the number, not proof that the model committed to it). Here is one of them:

**Question (GSM8K).** *Sally buys one small pizza at $8 and one family-size pizza that costs 3 times as much as the small pizza. How much does she spend in all?*

Base model, thinking mode. The response reaches the answer within its first ~300 tokens, then loops until the 4,096-token budget runs out, saying "Wait" 16 times. It never closes its reasoning, so it never gives a final answer. This is the last part of its thinking:

```text
    Let's write the response.
    Step 1: The small pizza costs $8.
    Step 2: The family-size pizza costs 3 times the small pizza, which is $3 \times 8 = 24$.
    Step 3: The total cost is $8 + 24 = 32$.
    \boxed{32}

    I will use \boxed{32}.
    Wait, I'll use \boxed{32}.
    Okay.

    Let's write the response.
    Step 1: The small pizza costs $8.
    ...
```

A (reward-only) closes its reasoning after 154 tokens and answers correctly:

```text
The user wants to calculate the total cost Sally spends.

1.  **Identify the cost of the small pizza:**
    *   Cost = $8.

2.  **Identify the cost of the family-size pizza:**
    *   It costs 3 times as much as the small pizza.
    *   Cost = 3 * $8 = $24.

3.  **Calculate the total cost:**
    *   Total = Cost of small pizza + Cost of family-size pizza.
    *   Total = $8 + $24 = $32.

4.  **Format the output:**
    *   Step-by-step explanation.
    *   Final answer in \boxed{}.
</think>

... Sally spends a total of $32.  \boxed{32}
```

## 10. Free baselines: brevity prompts backfire, no-thinking wins OOD {#controls}

[![Where the generated tokens go]({{ '/assets/research/thinkingcap-2b/figures/fig11_token_composition.png' | relative_url }})]({{ '/assets/research/thinkingcap-2b/figures/fig11_token_composition.png' | relative_url }})

*Figure 11. Mean generated tokens per ID question, split into the thinking span and the answer after `</think>`. The number at the end of each bar is the total.*

A token-efficiency result means little unless it beats the alternatives that cost nothing. We checked two.

**Asking for brevity backfires.** Prepending *"Answer using as few reasoning tokens as possible"* makes the base model think *longer*. Its thinking rises from 2,848 to 3,789 tokens, non-termination from 55% to 88.5%, and "wait" from 10.6 to 35.6 per response. ID accuracy collapses to 10.9%. The pathology is not a style choice that a prompt can remove.

**Switching thinking off is a strong baseline.** The untrained no-thinking model reaches 82.50% ID accuracy with **562 total tokens**, and it is the best condition on MATH-500 (71.0% against 58.5% for A). It still reasons, just in the answer: its answers average 561 tokens, against about 212 for A. A beats it in distribution by **+2.75 pp** [+0.25, +5.25], but at **63.1% more total tokens** [+48.5, +79.1] (916 against 562). So the precise claim is that **A is the most token-efficient *thinking* policy we found**, while no-thinking remains on the accuracy-versus-total-tokens Pareto front in distribution and dominates out of distribution at this scale.

## 11. Limitations {#limitations}

- **One seed per condition.** Our intervals resample prompts, not training runs, so they do not measure seed variance. The training seed alone can shift RL results: with identical hyperparameters, two groups of five seeds can produce statistically different learning curves ([Henderson et al., 2018](https://arxiv.org/abs/1709.06560)). The direction of each finding is consistent across benchmarks and training curves, but the size of every effect should be read as a single draw.
- **Greedy decoding, 4,096-token budget.** Qwen recommends sampling for thinking mode (temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5) and an output length of 32,768 tokens for most queries (81,920 for competition-level math and coding). The Qwen3 model cards (for example [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)) warn outright that greedy decoding in thinking mode *"can lead to performance degradation and endless repetitions."* Greedy decoding makes loops more likely, so the absolute non-termination rates describe this protocol, not the model's typical behavior. Greedy outputs were not even bit-identical across software stacks: re-running the same models on the same dev prompts moved the base model's non-termination rate from 58.3% in our first screening to 63.0% in the frozen evaluation, in line with the hardware- and software-dependent score shifts reported by [Hochlehnert et al. (2025)](https://arxiv.org/abs/2504.07086).
- **A missing free baseline.** We did not evaluate thinking mode with a forced close at the 4,096-token cap, as in s1-style budget forcing or Qwen3's thinking budget. Most of A's gain comes from curing non-termination, so that control would show how much of the gain a decode-time cap recovers at no training cost.
- **Framework confound.** A, +R, +R+D, and B use TRL; GDPO and ShortRL-G2 use verl. We never trained A under verl, so a follow-up's gap to A combines the algorithm with the whole training pipeline. Our plan also included BatchScale, a same-framework batch-normalization control that would have separated GDPO's batch re-standardization from its reward decomposition, and SerialExit-G3, a serial early-exit variant. Neither was run.
- **ShortRL-G2 is an adaptation, not a reproduction.** Short-RL uses eight rollouts, a bonus within a length tolerance, min–max length normalization over total length, and gating on batch accuracy. Our two-rollout version keeps the core idea (penalize only when accuracy is high, relative to the shortest correct answer) and changes most details.
- **Training budget versus evaluation budget.** Rollouts are capped at 1,280 tokens, while evaluation allows 4,096. A's run truncated 60% of all training rollouts (81% in the first 100 updates). In the GDPO run, 62% of rollouts were truncated and therefore produced no gradient. A larger budget or group size might change the picture.
- **Smaller issues.** Both multiple-choice parsers read the answer letter case-insensitively, so a sentence such as "the answer is a plant" would count as choice A. The evaluation parser takes the last explicit *Answer: X* in the answer region; the training-time parser is looser and takes the first letter within eight characters of "answer". The +R run crashed once and resumed from an earlier checkpoint, and the B run also resumed once.

## 12. Conclusion {#conclusion}

We set out to make a 2B reasoner think less, and found that the base model's real problem was not thinking too much but never stopping. A correctness-gated length penalty, placed only in the reward and optimized with stock GRPO, largely fixes that in distribution, within 200 LoRA updates on one consumer GPU: it cuts thinking by three quarters and doubles strict accuracy. None of the refinements we tried improved on it. For most of them, the two-rollout normalization effect of Section 8 is our best explanation of why.

The practical lesson is worth testing well beyond this study: **before adding a loss term, check what your advantage normalization already does to the signal you care about.** At small group sizes, the group-std "bias" may be the strongest length regularizer you have.

We would most like to see three follow-ups: multi-seed replication, a matched verl GRPO control for the follow-up methods, and the same ladder at G = 4 and G = 8, where the normalization stops being a sign function.


## Data, code, and training logs {#data}

Every figure is generated from the saved per-item evaluation records and from the public W&B histories. The plotted series are available as files:

- [`nine_conditions.csv`]({{ '/assets/research/thinkingcap-2b/data/nine_conditions.csv' | relative_url }}): the main results table, with per-benchmark closure rates and medians.
- [`items.csv`]({{ '/assets/research/thinkingcap-2b/data/items.csv' | relative_url }}): strict correctness, closure, and token counts for all 11,700 responses. It contains no generated text.
- [`paired_bootstrap.json`]({{ '/assets/research/thinkingcap-2b/data/paired_bootstrap.json' | relative_url }}): every paired comparison in this post, with its 95% interval.
- [`training_trl_wandb.csv`]({{ '/assets/research/thinkingcap-2b/data/training_trl_wandb.csv' | relative_url }}) and [`training_verl_wandb.csv`]({{ '/assets/research/thinkingcap-2b/data/training_verl_wandb.csv' | relative_url }}): the training curves exactly as plotted.
- [`wandb_history_verification.json`]({{ '/assets/research/thinkingcap-2b/data/wandb_history_verification.json' | relative_url }}): a reconciliation of the six W&B histories against the local training logs. 1,199 of the 1,200 logged updates match, and none of the 50,210 compared values differ.
- [`reflection.csv`]({{ '/assets/research/thinkingcap-2b/data/reflection.csv' | relative_url }}) and [`answer_found_gsm8k.json`]({{ '/assets/research/thinkingcap-2b/data/answer_found_gsm8k.json' | relative_url }}): the inputs to Figure 10.
- [`figure_provenance.json`]({{ '/assets/research/thinkingcap-2b/data/figure_provenance.json' | relative_url }}): SHA-256 hashes of every published data file, the W&B histories behind the training figures, and every figure.

**Training runs on W&B** ([project](https://wandb.ai/dongryeol/reducing-token)): [A](https://wandb.ai/dongryeol/reducing-token/runs/arm_a) · [+R](https://wandb.ai/dongryeol/reducing-token/runs/abl_r) · [+R+D](https://wandb.ai/dongryeol/reducing-token/runs/abl_dr) · [B](https://wandb.ai/dongryeol/reducing-token/runs/arm_b) · [GDPO](https://wandb.ai/dongryeol/reducing-token/runs/gdpo-verl-seed0) · [ShortRL-G2](https://wandb.ai/dongryeol/reducing-token/runs/short-rl-g2-verl-seed0)

**Code:** [github.com/DONGRYEOLLEE1/reducing-think-token](https://github.com/DONGRYEOLLEE1/reducing-think-token)

**Model:** [drlee1/ThinkingCap-Qwen3.5-2B](https://huggingface.co/drlee1/ThinkingCap-Qwen3.5-2B) is the loss-level variant (B) released after the first screening. The strict numbers in this post supersede the screening numbers on its model card. The reward-only adapter (A), the most token-efficient thinking policy in this post, is not released yet.

## Citation {#citation}

```bibtex
@misc{lee2026thinkingcap2b,
  title  = {Teaching a 2B Reasoner When to Stop Thinking},
  author = {Lee, Dong-Ryeol},
  year   = {2026},
  url    = {https://dongryeollee1.github.io/research/thinkingcap-2b/}
}
```


## References {#references}

- Aggarwal, P., & Welleck, S. (2025). [L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning](https://arxiv.org/abs/2503.04697). COLM 2025.
- Arora, D., & Zanette, A. (2025). [Training Language Models to Reason Efficiently](https://arxiv.org/abs/2502.04463). NeurIPS 2025.
- BottleCap AI (2026). [Introducing efficiency focused "ThinkingCap" model series](https://bottlecapai.com/thinkingcap-qwen3-6-27b). BottleCap AI blog, 13 July 2026.
- Chen, X., et al. (2024). [Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs](https://arxiv.org/abs/2412.21187). arXiv:2412.21187.
- Clark, P., et al. (2018). [Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457). arXiv:1803.05457.
- Cobbe, K., et al. (2021). [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168). arXiv:2110.14168.
- Dang, Q.-A., & Ngo, C. (2025). [Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't](https://arxiv.org/abs/2503.16219). arXiv:2503.16219.
- Feng, S., et al. (2025). [Efficient Reasoning Models: A Survey](https://arxiv.org/abs/2504.10903). TMLR.
- Guo, D., et al. (DeepSeek-AI) (2025). [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948). arXiv:2501.12948; journal version in *Nature* 645.
- Han, T., et al. (2024). [Token-Budget-Aware LLM Reasoning](https://arxiv.org/abs/2412.18547). Findings of ACL 2025.
- Henderson, P., et al. (2018). [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560). AAAI 2018.
- Hendrycks, D., et al. (2021). [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874). NeurIPS 2021.
- Hochlehnert, A., et al. (2025). [A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility](https://arxiv.org/abs/2504.07086). COLM 2025.
- Hou, B., et al. (2025). [ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning](https://arxiv.org/abs/2504.01296). arXiv:2504.01296.
- Hu, E. J., et al. (2022). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR 2022.
- Kimi Team (2025). [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599). arXiv:2501.12599.
- Lasocki, K., et al. (BottleCap AI) (2026). [ThinkingCap-Qwen3.6-27B](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B). Hugging Face model card.
- Li, G., et al. (2025). [DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](https://arxiv.org/abs/2510.04474). ICLR 2026.
- Lightman, H., et al. (2023). [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050). arXiv:2305.20050.
- Liu, S.-Y., et al. (2025). [DLER: Doing Length pEnalty Right - Incentivizing More Intelligence per Token via Reinforcement Learning](https://arxiv.org/abs/2510.15110). arXiv:2510.15110.
- Liu, S.-Y., et al. (2026). [GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization](https://arxiv.org/abs/2601.05242). arXiv:2601.05242.
- Liu, Z., et al. (2025). [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783). COLM 2025.
- Luccioni, S., & Gamazaychikov, B. (2025). [AI Energy Score v2: Refreshed Leaderboard, now with Reasoning](https://huggingface.co/blog/sasha/ai-energy-score-v2). Hugging Face blog.
- Ma, W., et al. (2025). [Reasoning Models Can Be Effective Without Thinking](https://arxiv.org/abs/2504.09858). arXiv:2504.09858.
- Muennighoff, N., et al. (2025). [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393). arXiv:2501.19393.
- Ning, Y., et al. (2026). [HRBench: Benchmarking and Understanding Thinking-Mode Switch Strategies in Hybrid-Reasoning LLMs](https://arxiv.org/abs/2605.28398). arXiv:2605.28398.
- Pipis, C., et al. (2025). [Wait, Wait, Wait... Why Do Reasoning Models Loop?](https://arxiv.org/abs/2512.12895). ICML 2026.
- Qwen Team (2025). [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). Hugging Face model card.
- Qwen Team (2026). [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B). Hugging Face model card.
- Renze, M., & Guven, E. (2024). [The Benefits of a Concise Chain of Thought on Problem-Solving in Large Language Models](https://arxiv.org/abs/2401.05618). FLLM 2024.
- Schulman, J., et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). arXiv:1707.06347.
- Schulman, J., & Thinking Machines Lab (2025). [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/). Thinking Machines Lab blog.
- Shah, V., et al. (2025). [A Comedy of Estimators: On KL Regularization in RL Training of LLMs](https://arxiv.org/abs/2512.21852). arXiv:2512.21852.
- Shao, Z., et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). arXiv:2402.03300.
- Sheng, G., et al. (2024). [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256). EuroSys 2025.
- Sui, Y., et al. (2025). [Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models](https://arxiv.org/abs/2503.16419). TMLR.
- von Werra, L., et al. (2020). [TRL: Transformers Reinforcement Learning](https://github.com/huggingface/trl). GitHub repository.
- Wang, S., et al. (2025a). [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2506.01939). NeurIPS 2025.
- Wang, S., et al. (2025b). [Tina: Tiny Reasoning Models via LoRA](https://arxiv.org/abs/2504.15777). arXiv:2504.15777.
- Wu, Y., et al. (2025a). [It Takes Two: Your GRPO Is Secretly DPO](https://arxiv.org/abs/2510.00977). arXiv:2510.00977.
- Wu, Y., et al. (2025b). [When More is Less: Understanding Chain-of-Thought Length in LLMs](https://arxiv.org/abs/2502.07266). arXiv:2502.07266.
- Xu, S., et al. (2025). [Chain of Draft: Thinking Faster by Writing Less](https://arxiv.org/abs/2502.18600). arXiv:2502.18600.
- Yang, A., et al. (2025). [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388). arXiv:2505.09388.
- Yu, Q., et al. (2025). [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476). arXiv:2503.14476.
- Yuan, D., et al. (2025). [Shorten After You're Right: Lazy Length Penalties for Reasoning RL](https://arxiv.org/abs/2505.12284) (first released as *Efficient RL Training for Reasoning Models via Length-Aware Optimization*). Findings of ACL 2026.

*The presentation of this write-up follows the structure of the [UltraBERT](https://huggingface.co/blog/dmis-lab/ultrabert) research blog.*

<style>
  mjx-container[display="true"] { overflow-x: auto; overflow-y: hidden; max-width: 100%; }
  /* compact cells so the nine-column results table fits the ~650 px post column */
  table th, table td { padding: 6px 8px; }
</style>

<script>
  window.MathJax = {
    tex: { inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']] },
    options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }
  };
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
