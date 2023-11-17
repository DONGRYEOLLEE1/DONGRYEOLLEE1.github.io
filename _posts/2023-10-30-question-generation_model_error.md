---
layout: post
title: CUDA error - device-side assert triggered Compile with TORCH_USE_CUDA_DSA to enable device-side assertions.
subtitle: 
tags: [python, Question-Generation]
categories: NLP
use_math: true
comments: true
published: true
---

# Env

- OS :  Ubuntu 22.04
- python : 3.10.12
- transformers : 4.35.2
- colab

# Error

- `(ko)bart` 베이스모델에서 tokenization -> model generation 할때 오류 발생

```python
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import torch

model_id = 'Sehong/kobart-QuestionGeneration'
tokenizer = PreTrainedTokenizerFast.from_pretrained(qg_kor_model_id)
model = BartForConditionalGeneration.from_pretrained(qg_kor_model_id, num_labels = 2).to('cuda')

text = """
부실 운영 논란을 빚는 2023 새만금 세계스카우트 잼버리 대회에 참가한 모든 참가자들이 7일 조기 퇴영하기로 했다.
세계스카우트연맹은 이날 오후 2시께 연맹 공식 누리집에 공지를 올려 “오늘 오전 한국 정부로부터 태풍 ‘카눈’의 영향이 예상됨에 따라 모든 잼버리 참가자들이 새만금에서 조기 퇴영할 계획이라는 확인을 받았다”고 밝혔다.
이어 “한국 정부가 모든 참가자들의 퇴영 일정과 장소에 대한 세부 정보를 제공할 것이라고 알려왔다”며 “모든 참가자들에게 본국으로 돌아가기 전 (한국에) 체류하는 동안 필요한 지원을 제공할 것을 긴급히 요청한다”고 덧붙였다.

잼버리 대원들은 서울 등 수도권으로 이동해 남은 일정을 보낼 것으로 보인다.
이날 오전 윤석열 대통령은 한덕수 국무총리와 이상민 행정안전부 장관에게 태풍 대비 잼버리 컨틴전시 플랜을 보고받고 점검했다고 김은혜 대통령실 홍보수석이 서면 브리핑에서 밝혔다.
김 수석은 “‘컨틴전시 플랜’이란 스카우트 대원들의 숙소와 남은 일정이 서울 등 수도권으로 이동할 수 있음을 의미한다”고 설명했다.
""" * 10

print(len(text))
>>> 5320
```

```python
from tqdm import tqdm
for i in tqdm(range(50), desc = 'test'):
    raw_input_ids = qg_kor_tokenizer.encode(text,)
    input_ids = [qg_kor_tokenizer.bos_token_id] + raw_input_ids + [qg_kor_tokenizer.eos_token_id]
    que_ids = qg_kor_model.generate(torch.tensor([input_ids]).to('cuda'), max_length = 4000)
    decode = qg_kor_tokenizer.decode(que_ids.squeeze().tolist(), skip_special_tokens = True)
```

```python
RuntimeError                              Traceback (most recent call last)
<ipython-input-9-fbaaf4cc870b> in <cell line: 2>()
      3     raw_input_ids = qg_kor_tokenizer.encode(text,)
      4     input_ids = [qg_kor_tokenizer.bos_token_id] + raw_input_ids + [qg_kor_tokenizer.eos_token_id]
----> 5     que_ids = qg_kor_model.generate(torch.tensor([input_ids]).to('cuda'), max_length = 4000)
      6     decode = qg_kor_tokenizer.decode(que_ids.squeeze().tolist(), skip_special_tokens = True)

RuntimeError: CUDA error: device-side assert triggered Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

# Solution

```python
/usr/local/lib/python3.10/dist-packages/torch/nn/functional.py in embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
   2231         # remove once script supports set_grad_enabled
   2232         _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
-> 2233     return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
   2234 
   2235 
```

- embedding 과정에서 dimension error로 추정
- `max_length = 1024` cut-off length 지정 & 추가해주기

```python
for i in tqdm(range(50), desc = 'test'):
    raw_input_ids = qg_kor_tokenizer.encode(text, max_length = 1024)
    input_ids = [qg_kor_tokenizer.bos_token_id] + raw_input_ids + [qg_kor_tokenizer.eos_token_id]
    que_ids = qg_kor_model.generate(torch.tensor([input_ids]).to('cuda'), max_length = 4000)
    decode = qg_kor_tokenizer.decode(que_ids.squeeze().tolist(), skip_special_tokens = True)
```

```python
>>> test: 100%|██████████| 50/50 [00:15<00:00,  3.15it/s]
```