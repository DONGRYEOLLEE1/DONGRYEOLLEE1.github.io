---
layout: post
title: TypeError - TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
subtitle: 
tags: [Finetuning, LLM]
categories: Developing
use_math: true
comments: true
published: true
---

## Error

```script
TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
```

## 원인

- Data-Augmentation을 위해 데이터셋 Re-generation 후, 학습 모듈 실행 도중 에러 발생
- **tokenization 과정**에서 오류 발생
- 확인 결과, Data-Augmentation하면서 특정 category의 데이터에 **missing values**가 발생되어 해당 오류 발생하였음

## Solution

```python
df = pd.read_csv(...)
df = df.dropna(axis = 0)

df = DA(...)

train_sentence = tokenizer(list(train_data["text"]), return_tensors = 'pt', padding = True, truncation = True, add_special_tokens = True)
...
```