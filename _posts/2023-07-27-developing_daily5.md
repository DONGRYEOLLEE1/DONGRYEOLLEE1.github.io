---
layout: post
title: Daily of Developing [7/27]
subtitle: 정규표현식, QG, QA, Data-Preprocessing
tags: [python, Question-Generation]
categories: Developing
use_math: true
comments: true
published: true
---

# Data Processing

- TASK: 언어모델의 성능 향상을 위한 파인튜닝 데이터 re-preprocessing

## 정규표현식

- 특정 문자열의 형태를 반복하는 데이터 이후에 나오는 문장을 모두 삭제

> Vol.{1~3자리 수} p.{1~3자리 수}

- 문자열 `Vol` 다음에 `.` 온점과 숫자가 오고, 그 뒤에 공백이 있으며 그 다음에 `p`, `.` 온점과 숫자가 오는 패턴

```python
regexp = "Vol\.\d+\sp\.\d+"
regexp_list = re.findall(regexp, data)
try:
    data = data.split(regexp_list[0])[0].strip()
except IndexError as e:
    pass
```

- `\.` : 온점(.)을 매칭시키기 위해 백슬래시 사용
- `\d+` : 숫자 하나 이상을 매칭
- `\s` : 공백 문자 매칭

- 위 내용은 [ChatGPT-3.5](https://chat.openai.com/)를 통해 생성된 답변입니다.

## Generative Model

### Question-Generation (한국어)

#### Kobart-QuestionGeneration

- 현 QA downstream task 모델 중 가장 성능이 좋은 모델
- 유려한 문장을 생성하지만 문장 전체의 주제를 포괄하는 질문을 생성해내진 못함
- [Huggingface](https://huggingface.co/Sehong/kobart-QuestionGeneration)
- [Github](https://github.com/Seoneun/KoBART-Question-Generation)


### Question-Generation (영어)

- 영어로된 데이터셋을 최초에 번역한 후에 한국어 QG 모델에 통과시키려 했으나 번역이 온전치 않기에 QG모델 -> 번역 순으로 전처리 진행

#### patil-suraj/question_generation

- requirements: `Transformers==3.0.0`, `nltk`, `nlp==0.2.0`
- 모델 크기별 QA, QG 모델이 각가 존재. 혼합 모델도 있음
- QA, QG 태스트 모두 답변 가능하며 다수의 문장을 뱉어주나 성능 면에서는 좋지 못한 것으로 판단되며 `transformers` 버전 이슈로 인해 가상환경을 따로 구축해줘야해서 사용하기 까다로움

| Name                                                                       | BLEU-4  | METEOR  | ROUGE-L | QA-EM  | QA-F1  | QG-FORMAT |
|----------------------------------------------------------------------------|---------|---------|---------|--------|--------|-----------|
| [t5-base-qg-hl](https://huggingface.co/valhalla/t5-base-qg-hl)             | 21.3226 | 27.0854 | 43.5962 | -      | -      | highlight |
| [t5-base-qa-qg-hl](https://huggingface.co/valhalla/t5-base-qa-qg-hl)       | 21.0141 | 26.9113 | 43.2484 | 82.46  | 90.272 | highlight |
| [t5-small-qa-qg-hl](https://huggingface.co/valhalla/t5-small-qa-qg-hl)     | 18.9872 | 25.2217 | 40.7893 | 76.121 | 84.904 | highlight |
| [t5-small-qg-hl](https://huggingface.co/valhalla/t5-small-qg-hl)           | 18.5921 | 24.9915 | 40.1886 | -      | -      | highlight |
| [t5-small-qg-prepend](https://huggingface.co/valhalla/t5-small-qg-prepend) | 18.2791 | 24.6722 | 39.958  | -      | -      | prepend   |


```python
!pip install transformers==3.0.0 nltk nlp==0.2.0
!python -m nltk.downloader punkt
!git clone https://github.com/patil-suraj/question_generation.git
%cd question_generation

from pipelines import pipeline

data = "..."

nlp = pipeline("question-generation", model="valhalla/t5-small-qg-prepend", qg_format="prepend")
nlp(data)

"""
[{'answer': '<pad> How long can this trend really be sustained',
 'question': 'What do you want to ask yourself to prove why the shape of the future  should be fundamentally different from the more cyclical past?'},
{'answer': '<pad> Challenge yourself to try and prove why the shape of the future should be so fundamentally different from the more cyclical past',
'question': 'How long can this trend really be sustained?'},
...
"""
```

#### T5-end2end-question-generation

- 영어모델중에서 가장 높은 성능이라 판단하며 `output`으로 여러 문장이 생성
- `Futher Works`: `output`으로 생성된 문장과 원데이터를 유사도 분석하여 similarity가 가장 높은 문장을 최종 채택할 수 있음
- Similarity method
  - `cosine similarity`

```python
from transformers import T5ForConditionalGeneration, T5TokenizerFast


checkpoint = "t5-base"
tokenizer = T5TokenizerFast.from_pretrained(checkpoint)
hfmodel = T5ForConditionalGeneration.from_pretrained("ThomasSimonini/t5-end2end-question-generation")

tokenizer.sep_token = '<sep>'
tokenizer.add_tokens(['<sep>'])

def hf_run_model(input_string, **generator_args):
  generator_args = {
  "max_length": 256,
  "num_beams": 4,
  "length_penalty": 1.5,
  "no_repeat_ngram_size": 3,
  "early_stopping": True,
  }
  input_string = "generate questions: " + input_string + " </s>"
  input_ids = tokenizer.encode(input_string, return_tensors="pt")
  res = hfmodel.generate(input_ids, **generator_args)
  output = tokenizer.batch_decode(res, skip_special_tokens=True)
  output = [item.split("<sep>") for item in output]
  return output

t5_output = hf_run_model(data1)
t5_output

"""
[["What are scenarios a powerful tool in the strategist's armory?",
  ' What do scenarios enable the strategist to steer a course between?',
  ' How many features make them a particularly powerful tool for understanding uncertainty and developing strategy accordingly?',
  '']]
"""
```

#### Sehong/t5-large-QuestionGeneration

- `t5-end2end-question-generation` 모델과 비교될 정도로 좋은 모델이나 1개의 문장만을 생성하며 원데이터 앞에 special token을 prepend 해줘야 좋은 문장을 뱉어줌
  - special token: `answer:`
- 생성된 문장에 앞단에 `question:` token을 prepend된 상태로 나옴


```python
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained('Sehong/t5-large-QuestionGeneration')
model = T5ForConditionalGeneration.from_pretrained('Sehong/t5-large-QuestionGeneration').to('cuda')

text = "..."
text_append = "answer:" + text

raw_input_ids_ = tokenizer.encode(data_append)
input_ids_ = [tokenizer.bos_token_id] + raw_input_ids_ + [tokenizer.eos_token_id]
question_ids_ = model.generate(torch.tensor([input_ids_]).cuda(), num_beams = 4, max_length = 512, eos_token_id = 1)
decode_ = tokenizer.decode(question_ids_.squeeze().tolist(), skip_special_tokens = True)

decode_

>>> question: What did he do to avoid looking stupid during the financial crisis?
```