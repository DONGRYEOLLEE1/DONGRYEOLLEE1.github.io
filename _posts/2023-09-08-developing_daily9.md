---
layout: post
title: Pytorch에서 CUDA 및 그래픽카드 인식 문제 - Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error?
subtitle: 
tags: [Pytorch, CUDA, Fabricmanager, Ubuntu]
categories: Developing
use_math: true
comments: true
published: true
---

# Env

- OS :  Ubuntu 22.04
- GPU : Nvidia A100 * 8
- nvidia-driver : 525.125.06
- cuda : 11.8
- pytorch : 2.1.0+cu118

## Issue

- `torch` 설치 후, 그래픽드라이버 checking하는 과정에서 cuda 및 그래픽드라이버 인식 ❌
- `nvcc -V` -> 11.8 정상적으로 출력
- 다른 사람들의 TS : 
    1. reboot
    2. nvidia driver 재설치
    3. cuda 또는 driver 버젼 mismatch -> 재설치

- 모두 시도해봤으나 해결 ❌

```
UserWarning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 804: forward compatibility was attempted on non supported HW (Triggered internally at  /opt/conda/conda-bld/pytorch_1603729096996/work/c10/cuda/CUDAFunctions.cpp:100.)
```

## 원인

- 검색 결과, `fabric-manager` 설치 안해서 인식 못한듯
- A100에서 나타는 이슈

## 해결

- `fabric-manager` 설치 : 그래픽드라이버 버전 맞춰서 설치

```
apt-get install cuda-drivers-fabricmanager-525
```

```python
# gpu_test.py
import torch

print("GPU 사용가능: ", torch.cuda.is_available())
print("GPU 이름: ", torch.cuda.get_device_name())
print("GPU 사용 개수: ", torch.cuda.device_count())

>>> GPU 사용가능:  True
>>> GPU 이름:  NVIDIA A100-SXM4-80GB
>>> GPU 사용 개수:  8
```

## Reference

- [apt-packaging-fabric-manager](https://github.com/NVIDIA/apt-packaging-fabric-manager)
- [CUDA initialization: Unexpected error from cudaGetDeviceCount()](https://stackoverflow.com/questions/66371130/cuda-initialization-unexpected-error-from-cudagetdevicecount)