---
layout: post
title: Daily of Developing [8/1]
subtitle: Finetuning, QG, Translate
tags: [Finetuning]
categories: Finetuning
use_math: true
comments: true
published: true
---

## Workflow

![workflow](/img/dataset_generation_workflow.png)

- [SelFee](https://kaistai.github.io/SelFee/) 방식에 영감을 받아 Self-Feedback method 적용

## Model

### Question-Generation(eng)

- [ThomasSimonini/t5-end2end-question-generation](https://huggingface.co/ThomasSimonini/t5-end2end-question-generation)

### Question-Generation(kor)

- [Sehong/kobart-QuestionGeneration](https://huggingface.co/Sehong/kobart-QuestionGeneration)

### SBERT

- [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

### Translation

- 데이터 보안 이슈로 인해 API 사용 x
- 번역 특화 모델 사용
- [KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-bidirection](https://huggingface.co/KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-bidirection)