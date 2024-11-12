---
layout: post
title: (LLM-University) 1-3. Dense Retrieval
subtitle: cohere - LLM University
tags: [NLP, RAG]
categories: NLP
use_math: true
comments: true
published: true
---

## Dense Retrieval

- Who discovered penicillin?
- Who was the first person to win two Nobel prizes?

Keyword search를 통해 첫번째 query는 잘 작동했지만 두번째 query는 그렇지 못했습니다. 두번째 query에는  실제로 많은 document에 query에 있는 단어들이 불필요하게 많이 포함되어 있기 때문입니다. 좋은 결과를 얻어내는 방법으론 모델이 '잘 이해할 수 있는' 질문을 하는 것입니다. 바로 이 부분에서 Semantic Search가 기인합니다. Semantic Search는 단순한 키워드 매칭이 아닌 의미를 파악하여 검색합니다. 언어 모델들은 두 가지 주된 semantic search 방법론(Dense Retrieval & Re-ranking)을 활용합니다. 그 중 Dense Retrieval을 살펴봅시다.

### Querying the Dataset Using Dense Retrieval

Dense Retrieval은 query와 유사한 문서를 찾기위해 text embedding을 사용합니다. Embedding은 숫자로 된 긴 리스트 데이터인 vector를 텍스트의 각각(token)에 할당합니다. Embedding에서의 주요 속성 중 하나는 비슷한 텍스트 조각이 비슷한 벡터로 이동한다는 것입니다. 

간단히 말해서, dense retrieval은 다음으로 구성되어있습니다.

- query에 상응하는 embedding vector를 찾는것
- 각각의 응답에 상응하는 embedding vector를 찾는 것 (이번의 경우엔 Wikipedia article이 되겠음)
- 임베딩안에서 query vector와 가장 가까운 response vector을 검색하는 것

Dense retrieval을 구현하는 파이썬 코드는 아래와 같이 구성할 수 있으며 cohere api를 사용하였습니다.

```python
def dense_retrieval(query, results_lang='en', num_results=10):

    nearText = {"concepts": [query]}
    properties = ["text", "title", "url", "views", "lang", "_additional {distance}"]

    # To filter by language
    where_filter = {
        "path": ["lang"],
        "operator": "Equal",
        "valueString": results_lang
        }
    response = (
        client.query
        .get("Articles", properties)
        .with_near_text(nearText)
        .with_where(where_filter)
        .with_limit(num_results)
        .do()
    )

    result = response['data']['Get']['Articles']
    return result
```

### Chunking the Articles

이 프로세스는 embedding에서 query에 가장 가까운 문서를 찾는 방식으로 좋은 결과를 얻을 수 있습니다. 하지만, 문서가 매우 길 경우 복잡해질 수 있습니다. 더 세분화된 처리를 위해, 문서를 단락별로 나누겠습니다. 즉, 각 문서의 각 단락에 해당하는 embedding 벡터를 찾는다는 의미입니다. 이렇게 하면, 모델이 답을 검색할 때 query와 가장 유사한 단락과 해당 단락이 속한 문서를 실제로 출력할 수 있습니다.

- simple query 결과:
    - response:
        - Alexander Fleming: “Sir Alexander Fleming (6 August 1881 - 11 March 1995) was a Scottish physician and microbiologist …”
        - Penicillin: “Penicillin was discovered in 1928 by Scottish scientist Alexander Fleming …”
        - Penicillin: “The term “penicillin” is defined as the natural product of “Penicillium” mould with antimicrobial activity. It was coined by Alexander Fleming ...”

- hard query 결과:
    - response:
        - Nobel prize in literature: “The Nobel prize in literature can be shared by two individuals …”
        - Nobel prize: “Although posthumous nominations are not presently permitted, …”
        - Nobel prize: “Few people have received two Nobel prizes. Marie Curie received the Physics prize …”
        - Marie Curie: “Marie Curie was the first woman to win a Nobel prize, the first person to win two Nobel prizes, …”


확실히 keyword search 알고리즘보다 훨씬 더 좋은 결과를 나타냅니다. hard query의 response 중, 두 번째 그리고 네 번째 결과에서도 올바른 문서를 찾았고 실제로도 세 번째와 네 번째 결과는 정답의 근거가 되는 문장으로 보입니다. 이는 임베딩이 텍스트의 의미를 포착하여 두 텍스트가 공통된 단어를 많이 공유하지 않더라도 비슷한 의미를 가지고 있는지를 파악할 수 있기 때문입니다.