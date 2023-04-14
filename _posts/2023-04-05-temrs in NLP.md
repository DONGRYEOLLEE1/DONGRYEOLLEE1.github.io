---
layout: post
title: Terms in NLP
subtitle: Term
tags: [NLP, Term]
categories: NLP
use_math: true
comments: true
---

> 논문이나 각종 refernece에서 나타나는 용어들에 대해 정리한 글입니다. (자주 까먹어서..)



## `NLI - Natural Language Inference`

- 자연어 추론
- 두 개 문장(premise, hypothesis)이 참(entailment), 거짓(contradiction), 중립(netural)인지 가려내는 classification task

> e.g. '나 출근했어' + '난 백수야' = 거짓(contradiction)

- NLI 데이터셋 (corp. UpStage)
- 전제(premise)에 대한 가설(hypothesis)이 참(entailment)인지 거짓(contradiction)인지 중립 혹은 판단불가인지 정보가 레이블(gold label)로 주어짐

    - 전제(premise) : 이 영화 볼빠에 차라리 잠을 더 자는게 이득일듯
    - 가설(hypothesis) : 잠을 더 많이 잤다
    - 레이블(gold_label) : `contradiction`


    - 전제(premise) : 이 영화 볼빠에 차라리 잠을 더 자는게 이득일듯
    - 가설(hypothesis) : 영화의 등장인물은 참 멋있었다
    - 레이블(gold_label) : `neutral`

    - 전제(premise) : 63빌딩 근처에 나름 즐길거리가 많다
    - 가설(hypothesis) : 63빌딩 부근에서는 여러가지를 즐길수 있다
    - 레이블(gold_label) : `entailment`

> 한국어 데이터셋 : [KorNLI KorSTS](https://github.com/kakaobrain/kor-nlu-datasets)

<br>

##  `GLUE - General Language Understanding Evaluation`

- 사전 학습된 딥러닝 기반 언어 모델인 `ELMo`, `GPT-1`, `BERT` 모두 GLUE benchmark에서 최고의 성능을 보였음

<br>

## Task on benchmarks

![ex](https://hryang06.github.io/assets/images/post/nlp/nlu-ex.png)
출처 : [https://hryang06.github.io/nlp/NLP/#nlpnatural-language-processing-%EC%9E%90%EC%97%B0%EC%96%B4-%EC%B2%98%EB%A6%AC](https://hryang06.github.io/nlp/NLP/#nlpnatural-language-processing-%EC%9E%90%EC%97%B0%EC%96%B4-%EC%B2%98%EB%A6%AC)


## Perplexity 

- **외부 평가보다 조금 부정확할 수는 있어도 테스트 데이터에 대해서 빠르게 식으로 계산되는 더 간단한 평가 방법**
- 모델 내에서 자신의 성능을 수치화하여 결과를 내놓는 내부 평가에 해당되는 것
- 언어 모델을 평가하기 위한 내부 평가지표 (헷갈리는 정도)
- Lower Better
- PPL은 선택할 수 있는 가능한 경우의 수를 의미하는 분기계수(branching factor)
- PPL의 값이 낮으면 높은 정확도를 보이는 것이지 사람이 직접 느끼기에 좋은 언어 모델이라는 것을 의미하진 않음