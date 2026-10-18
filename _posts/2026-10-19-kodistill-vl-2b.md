---
layout: post
title: "KoDistill-VL-2B: Korean OCR Post-Training on a Single GPU"
subtitle: "Literal transcription, bilingual replay, and measured gains over Qwen3.5-2B"
date: 2026-10-19 00:00:00 +0900
permalink: /research/kodistill-vl-2b/
canonical_url: https://dongryeollee1.github.io/research/kodistill-vl-2b/
lang: en
tags: [VLM, OCR, Qwen, SFT, GRPO, research]
categories: Research
use_math: false
comments: true
published: true
excerpt: "A single-GPU Korean OCR study: 71.26% K-STR word accuracy versus 67.62% for Qwen3.5-2B, with bilingual benchmarks, real training curves, and transparent limitations."
---

Small vision-language models can answer a question about an image without reliably transcribing its Korean text. KoDistill-VL explores that gap: start with a bilingual Qwen3.5-2B fine-tune, add literal Korean text recognition, and measure both Korean gains and English changes.

We release [drlee1/kodistill-vl-2b](https://huggingface.co/drlee1/kodistill-vl-2b), a fully merged BF16 model. The experiment uses one RTX 5080 with 16 GB of VRAM and checkpoints every optimizer update so work can continue across workstation shutdowns.

## At a glance

- **Korean word recognition:** 67.62% → **71.26%** against the original Qwen checkpoint, **+3.64 percentage points**; paired scene-bootstrap 95% CI **[+3.24, +4.04] pp**.
- **Character transcription:** corpus CER falls from 32.14% to **15.00%**, and WER from 46.79% to **33.47%** on the known-reference subset.
- **English evaluation:** MMStar rises from 40.73% to **58.40%**, and TextVQA from 75.27% to **78.22%**, under the same project evaluation settings.
- **An important distinction:** these base-to-final gains include the earlier bilingual training. The OCR continuation alone adds **+3.45 pp** word accuracy over V2. We do not attribute every gain to the last SFT or GRPO stage.

![General benchmark comparison]({{ '/assets/research/kodistill-vl-2b/figures/benchmark_comparison.png' | relative_url }})

*All three models are measured under the same project protocol. The comparison is with our remeasured base, not the official Qwen leaderboard.*

## The question: aligning Korean QA and transcription

The earlier V2 model improved Korean visual question answering, but its K-STR word accuracy barely moved: 67.62% → 67.81%. Its training included synthetic Korean documents and signs, yet much of that supervision asked for a field or an answer rather than a complete transcription.

That suggested a concrete follow-up: make the output match the evaluation task. We changed the OCR supervision to the full visible text of a crop, diversified printed fonts and image degradations, and added real Korean scene-text training crops. This is a hypothesis behind the recipe, not an isolated causal finding: the data mix, continuation exposure, and OCR rewards changed together.

## The recipe

![Training recipe and selected exposure]({{ '/assets/research/kodistill-vl-2b/figures/training_recipe.png' | relative_url }})

### 1. Preserve the bilingual foundation

The lineage is **original Qwen3.5-2B → bilingual V2 → OCR continuation → final merged model**. V2 uses published teacher annotations and accepted API distillation responses alongside original supervised QA/reward labels. Public teacher identities are publisher-attributed metadata; the API teacher is recorded under the served alias `gpt-6-luna`. We do not claim verified frontier backend versions or promote teacher guesses into GRPO reward gold.

The V2 SFT pool has 91,688 rows and its GRPO pool 10,959 prompts. Both complete one scheduled pass, but development selection retains earlier checkpoints. The table below distinguishes completed budgets from the prefixes that actually influence the selected model.

| Stage | Full pool | Completed updates | Selected update | Selected input exposure |
|---|---:|---:|---:|---:|
| V2 SFT | 91,688 rows | 5,731 | 1,500 | 24,000 rows |
| V2 GRPO | 10,959 prompts | 2,740 | 2,250 | 9,000 prompts |
| OCR SFT | 33,371 rows | 2,086 | 1,800 | 28,800 rows |
| OCR GRPO | 2,683 prompts | 671 | 600 | 2,400 prompts |

### 2. Align OCR supervision with literal transcription

The OCR SFT pool contains **22,000 synthetic transcriptions**, **3,179 real scene-text transcriptions**, and **8,192 general QA replay rows**. Of the synthetic rows, 20,000 are Korean and 2,000 English. The generator uses 20 OFL font families: 17 for training and three held out for development. Its Korean training vocabulary covers 1,391 Hangul syllables. Procedural textures, perspective, rotation, resizing, blur, compression, shadows, and light occlusion add variation without changing the known text label. Printed brush/pen fonts are not a natural-handwriting dataset.

Real crops come from the CLL-STR authors' Korean MLT19 **training** distribution. After overlap quarantine, 3,974 training crops are split into 3,179 SFT and 795 GRPO examples. The 455 official validation crops remain development-only. Benchmark-image hashes and near-image matches are checked before training; upstream parent-scene IDs are unavailable, so this does **not** establish zero scene-level overlap.

English replay is deliberate. Short OCR targets would otherwise make row counts a poor proxy for language exposure. In the SFT pool, English contributes **33.68% of supervised label tokens**, and transcription contributes **53.51%**. General replay answers are length-filtered so long descriptions do not dominate the completion loss. No new paid API generation is needed for this continuation: synthetic images have rendered-text labels, and real images retain their original transcriptions.

### 3. Continue with SFT and verifiable GRPO

Both continuation stages use language-side rank-16 LoRA, alpha 32, and zero dropout. Vision weights, input embeddings, and the output head remain frozen. The implementation uses Transformers 5.19.0, PEFT 0.21.2, and TRL 1.15.0 with an atomic resume loop; it does not use verl or Unsloth for these reported runs.

| Setting | OCR SFT | OCR GRPO |
|---|---|---|
| Learning rate | 5e-6 | 1e-6 |
| Completed pool passes | 1 | 1 |
| Microbatch × accumulation | 8 × 2 | 4 × 4 generated sequences |
| Prompt groups per update | 16 supervised rows | 4 prompts × 4 responses |
| Image/context budget | 512 visual tokens / 2,048 tokens | Same |
| Completion limit | Within the SFT context | 256 tokens |
| KL coefficient | — | 0.04 |

GRPO uses an immutable reference made from **this run's selected SFT model**. For an OCR completion, reward is `0.8 × max(0, 1 − CER) + 0.2 × exact_match`. Empty or truncated completions receive zero. Gold and predictions use NFC normalization and outer trimming; case, numbers, punctuation, and internal spaces remain meaningful. General QA replay keeps its existing task-specific verifier and original reward labels.

Checkpoints commit the adapter, optimizer, scheduler, RNG, sampler cursor, and logging state after complete optimizer updates. A real shutdown interrupted final evaluation after 7,296 cached questions; the enabled service recovered them at the next boot, generated the remaining 1,502 questions, and finished K-STR. This verifies boot-time evaluation recovery. Training also resumed from committed steps after interruptions; a physical power cut during training was not separately tested.

## What the training curves say

![Logged training and independent development curves]({{ '/assets/research/kodistill-vl-2b/figures/training_dynamics.png' | relative_url }})

*Curves are reconstructed from the actual W&B event histories and development journals. Selected checkpoints are marked. Sanitized plotted series are included in [benchmark data]({{ '/assets/research/kodistill-vl-2b/data/benchmarks.csv' | relative_url }}), [training windows]({{ '/assets/research/kodistill-vl-2b/data/training_windows.csv' | relative_url }}), and [development data]({{ '/assets/research/kodistill-vl-2b/data/development.csv' | relative_url }}), with [SFT](https://wandb.ai/dongryeol/kodistill-vl/runs/04c5f64a) and [GRPO](https://wandb.ai/dongryeol/kodistill-vl/runs/8406868b) run links.*

Each stage has one logical W&B run, including resumptions. Independent reconciliation matches every event ID, step, metric hash, and metric value: 213 SFT events and 72 GRPO events. The curves preserve the logged variation and mark selected checkpoints on full training traces.

Checkpoint selection maximizes `0.7 × (1 − real Korean CER) + 0.3 × (1 − synthetic Korean CER)` on independent development data. It prefers candidates whose English TextVQA and A-OKVQA source scores each stay within **2 pp** of the V2 baseline; if none qualify, the best valid OCR candidate is retained with an explicit failed-guard flag. Both selected continuation checkpoints pass this development guard. K-STR and the five final benchmarks do not select checkpoints.

The English trajectory is not flat: the first SFT checkpoint loses 5 pp on development TextVQA before later candidates recover. A passing development margin is not evidence of no English decline everywhere.

The selected SFT model reaches **15.12% real Korean CER / 3.12% synthetic Korean CER**. Selected GRPO remains at **15.12% / 3.21%**. We therefore **do not find an additional GRPO gain on this development objective**. There is no full-test SFT-only ablation for the OCR continuation, so the final test gain cannot be partitioned into separate data, SFT, and GRPO effects.

## Final results

| Benchmark | Metric | Qwen3.5-2B | V2 | Final | Δ vs Qwen (pp) | Δ vs V2 (pp) |
|---|---|---:|---:|---:|---:|---:|
| K-MMStar | Accuracy ↑ | 44.07 | 53.73 | **54.20** | +10.13 | +0.47 |
| MTVQA-KO | Strict EM ↑ | 26.16 | 29.39 | **29.21** | +3.05 | -0.18 |
| K-DTCBench | Accuracy ↑ | 72.08 | 80.42 | **80.83** | +8.75 | +0.42 |
| MMStar (EN) | Accuracy ↑ | 40.73 | 58.40 | **58.40** | +17.67 | +0.00 |
| TextVQA (EN) | VQA accuracy ↑ | 75.27 | 78.12 | **78.22** | +2.95 | +0.10 |
| K-STR | Official word accuracy ↑ | 67.62 | 67.81 | **71.26** | +3.64 | +3.45 |
| K-STR | Jaso 1−NED ↑ | 87.12 | 87.13 | **89.08** | +1.96 | +1.95 |
| K-STR | CER ↓ | 32.14 | 32.81 | **15.00** | -17.14 | -17.81 |
| K-STR | WER ↓ | 46.79 | 44.85 | **33.47** | -13.32 | -11.37 |

Scores are percentages and changes are percentage points. The final column reports the continuation increment separately from the original-Qwen comparison.

### Korean OCR: a measurable change

![Korean OCR accuracy and transcription errors]({{ '/assets/research/kodistill-vl-2b/figures/korean_ocr_comparison.png' | relative_url }})

K-STR includes all **4,008 KAIST-Corrected** and **16,078 GIST-Signboard** public crops, for 20,086 total. Official word accuracy and Jaso 1−NED retain the upstream whitespace and `□` wildcard conventions. Auxiliary CER/WER exclude the 231 references with `□`, leaving **19,855** known-reference crops; CER retains internal spaces and is aggregated as total character edits divided by total reference characters.

The V2-to-final word-accuracy change is **+3.45 pp**, with a paired scene-bootstrap 95% interval of **[+3.04, +3.84] pp**. The original-Qwen-to-final interval is **[+3.24, +4.04] pp**. These intervals quantify resampling of this evaluation set, not training-seed variability; no multiple-comparison correction is applied.

CER drops substantially, but it is not interchangeable with word accuracy. Overlong or repeated wrong transcriptions can add many character insertions. The base, V2, and final K-STR outputs have 24, 22, and one truncated generations respectively, all retained in scoring. Better answer length and fewer runaway outputs can contribute alongside better text recognition; we do not attribute the CER reduction solely to glyph recognition.

### English: report the observed scope

Against the original Qwen checkpoint, English MMStar improves **+17.67 pp** and TextVQA **+2.95 pp**. Against V2, MMStar is unchanged and TextVQA improves **+0.10 pp**. These are observed scores on two benchmarks, not a general English non-inferiority claim. Korean MTVQA strict EM falls **−0.18 pp** from V2 even though the original-Qwen comparison remains positive.

## How the comparison was checked

All models use the same inputs and inference settings: BF16, non-thinking mode, greedy generation, batch size eight, a 512-image-token ceiling, and a 2,048-token context budget. General VQA allows 128 generated tokens for multiple choice and 64 for short answers; K-STR allows 256. Invalid/truncated answers retain their prescribed denominators. The short answer-format protocol differs from Qwen's official sampling recipe, so these numbers should not be compared directly with its model-card leaderboard.

The five VQA datasets contain 1,500 K-MMStar, 558 MTVQA-KO, 240 K-DTCBench, 1,500 MMStar, and 5,000 TextVQA questions. Every model has complete raw coverage. Historical base/V2 raw stores are identity-checked and rescored; only the final model requires new generation.

A separate audit executes the pinned upstream K-STR scoring functions on **all 20,086 final outputs**. All **40,172 per-item official metric values** (two metrics per output) and **21 aggregate values**, including independent CER/WER, match the report exactly. There are no missing/extra rows or timeouts; the final model's one truncated output is included. This is a new-output score audit, not 40,172 separate generations.

The final artifact contains 617 BF16 tensors in three safetensors shards, 2,213,241,664 serialized parameters, and no LoRA tensors or adapter configuration. The immutable model identity is `2151cdc44c095c649bc9942c3116a5e441b425681aa319c1102a980442e823a5`.

## Try the model

See the [model card](https://huggingface.co/drlee1/kodistill-vl-2b#use) for the tested Transformers example. For a small self-rendered demonstration image, use [the example Korean sign](https://huggingface.co/drlee1/kodistill-vl-2b/resolve/main/examples/korean_sign.png). It is an illustration, not an evaluation sample or an additional benchmark result.

## Limits and release scope

This is a single-seed, single-workstation study and a follow-up on public tests already examined during the project. Exact-image and near-image decontamination cannot prove scene-level independence when source grouping metadata is missing. Printed synthetic text does not establish natural handwriting capability. Neither causal GRPO improvement, broad multilingual preservation, nor state-of-the-art performance is established.

The release contains the final merged model, processor/configuration, figures, sanitized metric series, and research documentation. It excludes source photographs, fonts, and teacher-answer corpora. Our modifications and original research materials use the existing project's **noncommercial research terms**; the underlying Qwen model retains Apache-2.0. Third-party source conditions, including unresolved original-photo terms for the CLL-STR distribution, remain separate and are described in [release NOTICE](https://huggingface.co/drlee1/kodistill-vl-2b/blob/main/NOTICE).

## References

- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), the original post-trained vision-language checkpoint.
- [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer), the training framework behind the continuation objective.
- [CLL-STR](https://github.com/ku21fan/CLL-STR), the authors' preprocessed MLT19 Korean train/validation source.
- [K-STR-Bench](https://github.com/hoonisone/K-STR-Bench), the official Korean scene-text benchmark and scoring reference.
- [Google Fonts](https://github.com/google/fonts/tree/47d54ca1ebb14fd864695f1d39dc734f39017c33/ofl), the pinned OFL font source.
- [UltraBERT](https://huggingface.co/blog/dmis-lab/ultrabert), the presentation reference for this research write-up.


The final merged checkpoint and concise usage guide are available on [Hugging Face: drlee1/kodistill-vl-2b](https://huggingface.co/drlee1/kodistill-vl-2b).
