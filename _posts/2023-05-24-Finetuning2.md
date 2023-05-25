---
layout: post
title: Polyglot-ko model FT via LoRA 
subtitle: lora_target_modules
tags: [Finetuning, LoRA, Polyglot]
categories: NLP
use_math: true
comments: true
---

## Envs
- python3.10.6
- ubuntu 22.04
- pytorch 2.0.0+cu118
- transformers 4.28.1
- peft 0.3.0
- openai 0.27.4
- accelerate 0.18.0
- bitsandbytes 0.38.1

## Mission

- Polyglot-ko 모델을 LoRA로 Fine-tuning

- [finetune.py](https://github.com/tloen/alpaca-lora/blob/main/finetune.py) 코드 중, `model` 과 `tokenizer` load 부분을 polyglot 모델에 맞게 `AutoTokenizer`, `AutoModelForCausalLM`으로 바꿔준 상태.

![figure1](/img/FT/img_2.png)

## Prob

```
ValueError: Target modules [q_proj,v_proj] not found in the base model. Please check the target modules and try again.
```
- Hparams `--lora_target_moduels` 오류로 모델에 맞는 값을 적절하게 입력해주면 정상 작동


```python
# `finetune.py` line 44

def train(
    ...
    lora_target_modules: List[str] = [
        "query_key_value"
    ]
)
```


## Issue

- 1️⃣
```
compute capability < 7.5 detected! only slow 8-bit matmul is supported for your gpu!
```
> GPU capability 문제, 무시해도 되는 오류인듯

- 2️⃣
```python
# `finetune.py` line 118
tokenizer.pad_token_id = (
    0
)
```
> GPT-NeoX tokenizer의 pad_token_id는 2로 설정되어있는데 무시해도 될까?
> 학습 후, 모델의 성능을 보고 추후에 수정해야 될 듯

- 3️⃣
`adapter_model.bin` 파일의 용량이 443kb로 저장되었고 해당 이슈 조사 결과, peft의 버젼 문제로 인한 버그가 생긴듯함

```python
!pip install -q git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08
```

[Issue #317 참고](https://github.com/huggingface/peft/issues/317)

## Ref.

- [tloen/alpaca-lora](https://github.com/Beomi/KoAlpaca/issues/42)
- [peft Issue #40](https://github.com/huggingface/peft/issues/40)
- [tloen/alpaca-lora Issue #251](https://github.com/tloen/alpaca-lora/issues/251)
- [아카라이브 AI언어모델](https://arca.live/b/alpaca/75354696/361828881#c_361828881)
- [빵형의 개발도상국 - Alpaca LoRA 파인튜닝](https://www.youtube.com/watch?v=aUXwVp4eUH4&ab_channel=%EB%B9%B5%ED%98%95%EC%9D%98%EA%B0%9C%EB%B0%9C%EB%8F%84%EC%83%81%EA%B5%AD)


