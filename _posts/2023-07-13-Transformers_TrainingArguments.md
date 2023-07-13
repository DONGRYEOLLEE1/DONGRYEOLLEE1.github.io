---
layout: post
title: Transformers TrainingArguments' Hyperparameters
subtitle:
tags: [NLP, Finetuning]
categories: Finetuning
use_math: true
comments: true
published: true
---

하이퍼파라미터 인자 괄호 안의 문자는 차례대로 `형태`, `default value`를 의미합니다.

## 하이퍼파라미터

- `per_device_train_batch_size`(int, 8) : 학습시에 사용되는 각 머신당 batch size를 의미
- `gradient_accumulation_steps`(int, 1) : backward pass 및 update pass를 수행하기 전에 기울기를 누적할 업데이트 단계의 수
- `num_train_epochs`(float, 3.0) : epoch 
- `max_steps`(int, -1) : 양수로 지정하면 학습 steps의 수를 지정. `num_train_epochs`를 override.
- `warmup_steps`(int, 0) : 0부터 `learning_rate` 값까지의 linear warmup의 step number
- `logging_steps`(int, 500) : 평가 업데이트 단계 지정 수 
- `save_steps `(int, 500) : 두 개의 체크포인트가 저장되기 전 업데이트 단계 수
- `save_total_limit`(int) : 체크포인트의 총량을 제한. `output_dir`에서 이전 체크포인트를 삭제함
- `fp16`(bool, False): 32-bit 학습 or 15-bit(mixed) precision 학습
- `learning_rate`(float, 5e-5) : learning rate 설정 값

## Usage

```python
args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,
    max_steps = 100,
    learning_rate = 2e-8,
    fp16 = True,
    logging_steps = 20,
    output_dir = f"{OUTPUT_PATH}",
    optim = 'paged_adamw_8bit',
    save_total_limit = 3,
    ...
)
```

![Alt text](/img/image.png)

- 총 **100steps**만큼 수행하며 평가는 **20step을 지날때마다 출력**하며 OUTPUT_PATH에 저장되는 model 파일의 **checkpoint의 총량 수를 3개로 제한**

## ref

[docs_trainer_trainingarguments](https://huggingface.co/transformers/v3.0.2/main_classes/trainer.html#trainingarguments)