---
layout: post
title: Zero-Shot Learning in NLP
subtitle: Zero-Shot Learning
tags: [Zero-shot, NLP]
categories: NLP
use_math: true
comments: true
---


> 이 글은 [Zero-Shot Learning in Modern NLP](https://joeddav.github.io/blog/2020/05/29/ZSL.html)을 번역한 글이며 몇가지 설명을 추가하였습니다.


NLP는 최근 AI분야에서 뜨거운 감자입니다. 최근 커뮤니티는 인터넷에서 사용할 수 있는 방대한 양의 라벨링되지 않은 데이터에서 학습할 수 있는 꽤 효과적인 방법을 찾기 시작했습니다. 비지도학습 모델로부터의 전의학습의 성공은 우리를 사실상 존재하는 모든 downstream 지도학습 task를 능가하게 만들어줬습니다. 우리가 새로운 모델 또는 비지도학습 객체들을 개발하는것을 이어나가는것처럼 SOTA는 사용 가능한 엄청난 양의 레이블된 데이터를 이용한 많은 task를 통해 급격하게 그 목표를 수정하고 있습니다. 모델이 계속해서 성장해나감으로써 얻는 우리의 주된 메리트는 바로 downstream task에서의 엄청난 양의 레이블된 데이터에 대한 의존이 매우 느리게 감소하는 것입니다. [GPT-3 - Language Models Are Few-Shot Learners, 2020](https://arxiv.org/abs/2005.14165)는 downstream task에서의 가공할만한 성능을 더 적은 task-specific한 data와 함꼐 보여주고 있습니다. 이는 앞으로 더 작은 모델을 요하게 될 것입니다.

![GPT3 Performance](https://joeddav.github.io/blog/images/zsl/gpt3_triviahq.png)

그러나 이러한 모델의 크기는 현실(현업)에서 사용하기에는 조금 실용적이지가 않습니다. 예를 들자면, GPT-3 Large는 약 12개의 병렬처리된 GPU의 메모리에 아주 잘 적합하게 구성되었습니다. 실제로 annotated data는 매우 빈약하거나 전체적으로 사용불가합니다. BERT와 같은 GPT3보다 작은 모델들은 weight와 함께 엄청난 양을 encode하여 결과값을 배출하고 있습니다. 우리가 현명하다면 많은 작업별 annotated data 없이도 이러한 잠재 정보를 활용하는 방식으로 이러한 모델을 downstream task에 적용하는 몇 가지 기술을 알아낼 수 있을 것 같습니다.

이 글에선 SOTA 모델을 활용하여 annotated 학습 데이터셋 없이 sequence classification을 하는 방법을 소개하도록 하겠습니다.


## What is Zero-Shot Learning

전통적으로 ZSL은 대부분 상당히 특정한 유형의 작업을 가리키는 경우가 많았습니다. 1개의 셋에서 분류기를 학습한 후에 다른 셋에서의 평가하는 방식으로요. 최근에 특히 NLP 분야에서 모델이 명시적으로 학습되지 않은 작업을 수행하도록 한다는 의미로 훨씬 더 광범위하게 사용되고 있습니다. 잘 알다시피 GPT-2에서 저자들은 언어모델을  파인튜닝 없이 직접적으로 기계독해와 같은 downstream task에서 평가하였습니다. 

여기서 정의가 그다지 중요하지 않습니다만 어떤 개념들을 이해하는데 꽤나 효과적이고 다른 방법들을 비교할떄 실험에 대해서 이해하는 것을 잘 인지해야 합니다. 예를들어 기존의 ZSL 학습에서는 모델이 학습 데이터 없이도 해당 클래스를 예측할 수 있도록 보이지 않는 클래스에 대한 일종의 discriptor를 제공해야 합니다. 

