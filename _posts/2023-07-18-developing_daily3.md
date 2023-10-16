---
layout: post
title: Daily of Developing [7/20] - Repo id must be in the form 'repo_name' or 'namespace/repo_name'
subtitle: TGI, LocalLLM
tags: [NLP, Finetuning, Docker]
categories: Developing
use_math: true
comments: true
published: true
---

## Error

- HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'

## Solution

```
$ pwd
/mnt/huggingface
$ ls
opt-125m  version.txt
$ ls opt-125m/
config.json  generation_config.json  merges.txt  pytorch_model.bin  special_tokens_map.json  tokenizer_config.json  vocab.json


$ docker run --gpus all --shm-size 1g -p 8080:80 --volume /mnt/huggingface:/data ghcr.io/huggingface/text-generation-inference:latest --model-id /data/opt-125m --num-shard 1
```

```
cd workspace/model_file
$ docker run --gpus all --shm-size 1g -p 8081:80 --volume /home/ubuntu/workspace/model_file:/data ghcr.io/huggingface/text-generation-inference:latest --model-id /data/{YOUR_MODEL_DIR_NAME} --num-shard 2
```

## ref

- [#245](https://github.com/huggingface/text-generation-inference/issues/245)