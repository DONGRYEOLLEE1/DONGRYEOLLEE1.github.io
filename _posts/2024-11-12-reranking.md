---
layout: post
title: (LLM-University) 1-4. Reranking
subtitle: cohere - LLM University
tags: [NLP, RAG]
categories: NLP
use_math: true
comments: true
published: true
---

## Reranking


Reranking은 다음과 같이 작동합니다: 각 (쿼리, 응답) 쌍에 대해 관련성 점수를 부여합니다. 이름에서 암시하듯이, 응답이 쿼리와 관련성이 높은 경우 높은 점수를 부여하고, 그렇지 않은 경우 낮은 점수를 부여합니다. 이 장에서는 이 모듈에서 이전에 찾은 위키피디아 검색 결과를 개선하기 위해 재정렬을 사용하는 방법에 대해 배울 것입니다.

### Using Rerank to Improve Keyword Search

Reranking은 기존의 search system의 성능을 끌어올려주는 굉장한 방법론입니다. 짧게 말해서, reranking은 query와 response를 받고 그들끼리의 relevance score를 리턴합니다. 이를 통해 어떤 검색 시스템이든 활용하여 query에 대한 답을 포함할 가능성이 있는 여러 문서를 찾아낸 뒤, reranking 엔드포인트를 사용하여 이 문서들을 정렬할 수 있습니다.

![img1](https://cohere.com/_next/image?url=https%3A%2F%2Fcohere-ai.ghost.io%2Fcontent%2Fimages%2F2024%2F05%2F27c5174-image.png&w=1920&q=75)

- "Who was the first person to win two Nobel prizes?"에 대한 keyword search의 결과
    - Responses:
        - Neutrino
        - Western culture
        - Reality television

- "Who was the first person to win two Nobel prizes?"에 대한 keyword search의 top 20 결과
    - responses:
        - Neutrino
        - Western culture
        - Reality television
        - Peter Mullan
        - Indiana Pacers
        - William Regal
        - Nobel Prize
        - Nobel Prize
        - Nobel Prize
        - Noble gas
        - Nobel Prize in Literature
        - D.C. United
        - Nobel Prize in Literature
        - 2021-2022 Manchester United F.C. season
        - Nobel Prize
        - Nobel Prize
        - Zach LaVine
        - 2011 Formula One World Championship
        - 2021-2022 Manchester United F.C. season
        - Christians


마찬가지로 rerank endpoint를 호출하여 아래의 함수를 정의합니다. Input은 query와 responses이며 검색할 responses의 수를 지정해줍시다.

```python
def rerank_responses(query, responses, num_responses=3):
    reranked_responses = co.rerank(
        query = query,
        documents = responses,
        top_n = num_responses,
        model = 'rerank-english-v3.0',
        return_documents=True
    )
    return reranked_responses
```

Rerank는 relevance score와 함께 아래의 결과를 리턴합니다. 

- "Who was the first person to win two Nobel prizes?"에 대한 Reranking 결과
    - Responses:
        - Nobel Prize: “Five people have received two Nobel Prizes. Marie Curie received the …”        Relevance score: 1.00
        - Nobel Prize: “In terms of the most prestigious awards in STEM fields, only a small …”        Relevance score: 0.97
        - Nobel Prize in Literature: “There are also prizes for honouring the lifetime achievement of writers …”        Relevance score: 0.87


### 결론

Reranking은 특정 쿼리에 가장 관련성이 높은 응답을 찾는 데 매우 유용한 방법입니다. 이는 dense retrieval을 개선하기 위한 키워드 검색의 보완 방식으로도 유용합니다. 이번 포스트에서는 키워드 검색을 통해 답을 포함할 가능성이 있는 20개의 문서를 먼저 검색한 뒤, 재정렬 엔드포인트를 사용하여 그 중 상위 3개를 선택함으로써 키워드 검색 결과를 크게 향상시켰습니다.