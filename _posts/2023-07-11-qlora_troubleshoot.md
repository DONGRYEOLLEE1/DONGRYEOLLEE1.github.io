---
layout: post
title: QLoRA 파인튜닝 - trouble shooting 및 일지
subtitle: 
tags: [Finetuning, NLP, QLoRA]
categories: Finetuning
use_math: true
comments: true
published: true
---

# Trouble Shooting

## 1

### 현상
```
AttributeError: ~/qlora/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cpu.so: undefined symbol: cquantize_blockwise_fp16_fp4
```

### 해결
```
cd qlora/lib/python3.10/site-packages/bitsandbytes
cp libbitsandbytes_cuda118.so libbitsandbytes_cpu.so
```

cuda 버젼에 상응하는 파일을 복사(대체)해주면 된다

### ref

[TimDettmers/bitsandbytes#156 (comment)](https://github.com/TimDettmers/bitsandbytes/issues/156)

## 2


# 4bit quantization

- 환경 
  - ubuntu 20.04
  - python 3.10.6
  - accelerate 0.21.0.dev0
  - bitsandbytes 0.39.0
  - datasets 2.13.1
  - peft 0.4.0.dev0
  - scikit-learn 1.2.2
  - torch 2.0.1
- 사용 모델 : `polyglot-ko-12.8b`
- 필요 메모리 : `8GB`
- 코드 : [qlora.py](https://github.com/DONGRYEOLLEE1/Paper/blob/main/Learning/QLoRA/qlora_training.ipynb)

```markdown
Tue Jul 11 17:25:47 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-PCI...  On   | 00000000:00:06.0 Off |                    0 |
| N/A   32C    P0    37W / 250W |   7955MiB / 40960MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-PCI...  On   | 00000000:00:07.0 Off |                  Off |
| N/A   30C    P0    32W / 250W |      3MiB / 40960MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     55094      C   ...e/ubuntu/qlora/bin/python     7952MiB |
+-----------------------------------------------------------------------------+

# 4bit 적재 필요 메모리 7.9GB
```

```python
config = LoraConfig(
    r = 32,
    lora_alpha = 32,
    target_modules = ['query_key_value'],
    lora_dropout = .05,
    bias = "none",
    task_type = 'CAUSAL_LM'
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# trainable params: 26214400 || all params: 6628362240 || trainable%: 0.39548834313557374
```

```markdown
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-PCI...  On   | 00000000:00:06.0 Off |                    0 |
| N/A   32C    P0    37W / 250W |   9165MiB / 40960MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-PCI...  On   | 00000000:00:07.0 Off |                  Off |
| N/A   30C    P0    33W / 250W |      3MiB / 40960MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     74937      C   ...e/ubuntu/qlora/bin/python     9162MiB |
+-----------------------------------------------------------------------------+

# LoRA 적재 후, 9.1GB 
```

- learning setting
  - `lora_r` : 32
  - `per_device_train_batch_size` : 32
  - `gradient_accumulation_steps` : 32

- 결과 : 🚫OOM

- H-params re-setting
  - `lora_r` : 8
  - `per_device_train_batch_size` : 2
  - `gradient_accumulation_steps` : 8


```markdown
Wed Jul 12 17:25:58 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-PCI...  On   | 00000000:00:06.0 Off |                    0 |
| N/A   65C    P0   238W / 250W |  21923MiB / 40960MiB |     97%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-PCI...  On   | 00000000:00:07.0 Off |                  Off |
| N/A   33C    P0    55W / 250W |  11345MiB / 40960MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A    369247      C   ...e/ubuntu/qlora/bin/python    21792MiB |
|    1   N/A  N/A    369247      C   ...e/ubuntu/qlora/bin/python    11342MiB |
+-----------------------------------------------------------------------------+
```

## ✍🏼

- 파라미터를 낮추는 과정에서 발생한 🚫 memory 이슈
  1. 학습은 진행되나 초기 step(약 5-step) 진행시, OOM 메세지 출력
  2. 학습 진행하는 동안 GPU memory usage 상승되는 현상 발생