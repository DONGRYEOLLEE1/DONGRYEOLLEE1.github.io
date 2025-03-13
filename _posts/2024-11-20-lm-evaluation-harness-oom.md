---
layout: post
title: (vLLM) lm-evaluation-harness OOM
subtitle: s
tags: [NLP, RAG, vLLM]
categories: Developing
use_math: true
comments: true
published: true
---

## Env

- os: Ubuntu 22.04
- nvidia driver version: 535.183.01
- cuda: 12.4
- vllm version: 0.6.3.post1
- torch version: 2.4.0+cu124
- transformers version: 4.46.3

## Error

```bash
export MODEL_NAME="/data/models/gemma-2-27b-it"
export TAKS="..."

lm_eval --model vllm --model_args pretrained=$MODEL_NAME,tensor_parallel_size=2,gpu_memory_utilization=0.7,dtype=bfloat16 --output_path "results/gemma2-27b-it" --batch_size 2 --tasks $TASK
```

OOM 에러 발생!

## Solution

![img](/img/lm-eval-oom.png)

- `vllm` backend arguments에 `enforce_eager=True` 옵션 추가하기
- `batch_size` 줄여주기

```bash
lm_eval --model vllm --model_args pretrained=$MODEL_NAME,tensor_parallel_size=2,gpu_memory_utilization=0.7,dtype=bfloat16,enforce_eager=True --output_path "results/gemma2-27b-it" --batch_size 2 --tasks $TASK
```


## Reference

- [lm-evaluation-harness-#1923](https://github.com/EleutherAI/lm-evaluation-harness/issues/1923)