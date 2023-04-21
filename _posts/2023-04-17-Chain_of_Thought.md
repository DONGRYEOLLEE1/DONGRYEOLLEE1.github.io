---
layout: post
title: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
subtitle: CoT
tags: [CoT, NLP, Prompt Engineering]
categories: NLP
use_math: true
comments: true
---

## Introduction 

연구를 통해 2가지의 아이디어를 사용해 언어모델의 추론능력을 향상시킬 수 있는 방법을 도출해냈다.

1. 산술적 추론 기술은 자연어 생성의 이점을 누릴 수 있다. 근거를 생성하는 데 도움이 될 수 있다. 
2. LLM은 prompting을 통해 in-context FSL의 흥미로운 전망을 제공한다. 매 다른 task를 위한 model checkpoint (FT) 대신에 단조로운 prompt를 모델에 질의할 수 있음. 놀랍게도 단순한 QA task에서 성공적적인 실험을 거뒀음.

![ex1]('../../img/cot/ex1.png)

전통적인 ![FSL](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)은 추론 능력에서 형편없는 결과를 내놓았고 종종 모델 scale을 늘렸음에도 불구하고 성능이 올라가지 않았음. 이에 연구진은 위의 2가지 아이디어를 통해 그들의 한계점을 타파한 해결책을 제시하고 특히 추론 능력 task에서의 Few-shot prompting 능력에 대해 발견하고 prommpt를 `input`, `CoT`, `output` 총 3가지 요소로 구성하였음. 위는 그 예시임.

![figure2](../../img/cot/figure2.png)

arithmetic, commonsense, symbolic reasoning 총 3개의 benchmark에서 평가하였음. GSM8K benchmark 환경에서 PaLM 540B 모델을 사용한 결과 새로운 SOTA performance를 달성하였음. 이 prompting 능력의 접근법이 중요한 이유는 새로운 큰 데이터셋과 큰 모델의 checkpoint 없이 많은 task에서 자랑할만한 성능을 뽑아줌에 있음. 

## Chain-of-Thought Prompting

여러 단계를 거쳐야하는 수학 문제와 같은 복잡한 추론을 통해 문제를 푸는 절차를 연구진은 고려하였음. 이건 전형적인 문제를 하나하나 분해하여 정답에 접근하는 방식임. 이 논문의 목표는 LLM에 유사한 CoT를 생성할 수 있는 기능을 부여하는 것입니다. 일관된 일련의 중간 추론 단계를 생성할 수 있는 능력을 LLM에 부여하는 것입니다.

CoT 능력은 다음과 같은 이점이 존재합니다.

1. CoT는 LLM이 문제에 대해 여러 스텝을 통해 분해할 수 있게 합니다. 이는 더 많은 스텝을 요하는 문제에 대해 추가적인 연산을 가능케 합니다.
2. CoT는 모델에 설명가능한 window를 제공해줍니다. 
3. CoT는 많은 수학 문제, commonsense 추론문제, symbolic manipluation 그리고 잠재적으로 적용가능한 언어로된 모든 task에서 사용되어 질 수 있습니다.
4. 마지막으로 CoT 추론은 단순한 CoT 문장의 예시를 추가해줌으로써 모델을 매우 효율적으로 동작하게 해줍니다.

## Arithmetic Reasoning

Arithmetic reasoning에서 CoT prompting을 사용한 결과 540B의 파라미터를 가진 모델에서 몇몇의 다른 task-specific 파인튜닝한 모델과 견줄만한 성능을 뽑아냈고 심지어 GSM8K benchmark에선 SOTA를 달성하였음.

![figure3](../../img/cot/figure3.png)

### 