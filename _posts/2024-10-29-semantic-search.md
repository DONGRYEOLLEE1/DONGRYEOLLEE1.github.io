---
layout: post
title: [LLM-University] 1-1. What is Semantic Search?
subtitle: cohere - LLM University
tags: [NLP, RAG]
categories: NLP
use_math: true
comments: true
published: true
---

## What is Not Semantic Search?

Semantic Search 이전, 가장 유명한 검색 방법은 keyword search 였습니다. 다음의 예를 봅시다.

Query: Where is the world cup?

Response:
1. The world cup is in Qatar.
2. The sky is blue.
3. The bear lives in the woods.
4. An apple is a fruit.

Keyword search 결과는 다음과 같을 것입니다.

Response:
1. The world cup is in Qatar. (4 words in common)
2. The sky is blue. (2 words in common)
3. The bear lives in the woods. (2 words in common)
4. An apple is a fruit. (1 words in common)

위 query의 경우 1번의 문장이 keyword search의 결과, 가장 유사한 문장으로 나타날 것입니다. 그런데 다른 경우를 한 번 살펴봅시다.

1. Where in the world is my cup of coffee? (5 words in common)

위 문장은 query에 대해 5 단어가 매칭됩니다만 안타깝게도 위 문장은 query에 대한 response로써 적절치 않은 문장으로 보입니다.

stop words를 지정하여 keyword search를 개선할 수도 있으며 또한 TF-IDF 알고리즘을 통해 관계없는 단어로부터 관계있는 부분을 추출할 수도 있습니다. 그러나 언어적 특성상 keyword search는 가장 최상의 response를 찾는대는 실패할 수도 있을것입니다. 그렇기에 그 다음 알고리즘인 Semantic Search를 살펴봅시다.

Semantic Search는 다음의 내용을 따릅니다.
- 단어를 vector 정보로 나타내기 위해 **Text Embedding**을 사용합니다.
- **유사도**를 사용하여 vector간에 query와 가장 비슷한 데이터를 찾습니다.

## How to search Using Text Embedding?

임베딩은 각 문장에 숫자 목록인 벡터를 할당하는 방식입니다. Embedding의 가장 중요한 요소는 바로 텍스트 조각에 더 비슷한 vetor를 할당하는 것입니다. 예를들어 "Hello, how are you?" 문장과 "Hi, what's up?" 문장을 매우 유사한 수의 리스트 데이터가 할당될 것인데 반해, "Tomorrow is Friday" 문장과는 이전 문장과 다르게 꽤 다른 수의 리스트 데이터가 할당될 것입니다. 

![emb1](https://cohere.com/_next/image?url=https%3A%2F%2Fcohere-ai.ghost.io%2Fcontent%2Fimages%2F2023%2F02%2FVis-1-1.png&w=1920&q=75)

더 나아가 real-life에서 사용할법한 text embedding을 작은 데이터셋에 사용해봅시다. 주어진 query는 4개, 그에 상응하는 문장도 4개 입니다.

Queries:
- Where does the bear live?
- Where is the world cup?
- What color is the sky?
- What is an apple?

Responses:
- The bear lives in the woods
- The world cup is in Qatar
- The sky is blue
- An apple is a fruit

Cohere text-embedding 모델을 사용하여 1024 context-length를 가지는 8개의 vector 데이터를 만들 수 있습니다. 더불어서 dimensionality reduction algorithms을 사용하여 2개의 길이를 가지는 데이터로 변환할 수 있습니다. 이 데이터를 2개의 좌표를 통해 아래와 같이 plot을 그릴수 있습니다.

![plot-coord](https://cohere.com/_next/image?url=https%3A%2F%2Fcohere-ai.ghost.io%2Fcontent%2Fimages%2F2024%2F10%2Fd0c031b-image.png&w=1920&q=75)

위 plot은 Euclidean distance 알고리즘을 통해 결과물을 산출하였습니다만, 이는 각각의 text를 비교하기에 이상적인 방법은 아닙니다. 다음으로 가장 일반적인 방법론을 알아봅시다.

## Using Similarity to Find the Best Document

유사도의 수는 각 document의 쌍에 다음과 같은 요건으로 인해 할당된 숫자입니다.

- 텍스트와 텍스트 자체의 유사도는 매우 높은 수치이다.
- 2개의 매우 유사한 텍스트 조각 사이의 유사도는 높은 수치이다.
- 2개의 다른 텍스트 조각사이의 유사도는 매우 적은 수치이다.

이번 시간에는 **cosine similarity**를 사용할 것이며, 이는 0과 1사이의 값을 가집니다. 이 유사도는 텍스트와 동일한 텍스트는 항상 1의 값을 가지며 가장 낮은 값으로는 0의 유사도 값을 가집니다. 이제 semantic search를 사용하기위해, query와 모든 각각의 문장과의 유사도, 그리고 가장 높은 유사도값을 가지는 문장을 구해야합니다. 아래는 8개의 문장사이에 가장 높은 코사인 유사도 값을 그린 plot입니다.

![plot-c](https://cohere.com/_next/image?url=https%3A%2F%2Fcohere-ai.ghost.io%2Fcontent%2Fimages%2F2024%2F10%2F5117351-image.png&w=1920&q=75)

위 plot을 통해 나타나는 insight는 다음과 같습니다.

- 숫자(유사도)의 최대값은 1이다.
- 각 문장과 그에 상응하는 response 사이의 유사도값들은 0.7 정도이다.
- 다른 pair의 문장간의 유사도값들은 0.7보다 낮다.

예를 들어 “사과란 무엇인가요?”라는 쿼리에 대한 답을 검색하는 경우, 시맨틱 검색은 표의 두 번째 행을 살펴보고 가장 가까운 문장이 “사과란 무엇인가요?”라는 것을 알 수 있습니다. (유사도 1), “사과는 과일이다”(유사도 약 0.7)입니다.

## Multilingual Search

Embedding 모델은 지원하는 언어로 최대길이 1024를 가지는 각각의 텍스트 조각을 벡터 데이터로 전달할 것입니다. 텍스트의 유사한 조각은 유사한 벡터로 전달될 것입니다. 그러므로 어떤 언어로든 쿼리를 사용하여 검색할 수 있으며 모델은 다른 모든 언어로 답변을 검색합니다. 

![multi-emb](https://cohere.com/_next/image?url=https%3A%2F%2Fcohere-ai.ghost.io%2Fcontent%2Fimages%2F2024%2F10%2Fimage-626751.png&w=1920&q=75)


## Are Embeddings and Similarity Enough? (No)

과연 위에서 본 알고리즘을 통해 real-data에 대해 유연한 검색이 가능할까요? 그 대답을 절대적으로 "No"일 것입니다. 그에 대한 단적인 예를 아래의 plot으로 대체하고 글을 줄입니다.

Query: Where is the world cup?

Responses:
- The world cup is in Qatar
- The world cup is in the moon
- The previous world cup was in Russia

![plot-m](https://cohere.com/_next/image?url=https%3A%2F%2Fcohere-ai.ghost.io%2Fcontent%2Fimages%2F2024%2F10%2Faed2670-image.png&w=1920&q=75)


