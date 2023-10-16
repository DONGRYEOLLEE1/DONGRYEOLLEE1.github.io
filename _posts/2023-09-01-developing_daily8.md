---
layout: post
title: 그래픽드라이버 버젼 이슈
subtitle: 
tags: [Nvidia, Ubuntu]
categories: Developing
use_math: true
comments: true
published: true
---

# Env

- Ubuntu 22.04
- Nvidia A100 * 8
- nvidia-driver : 535.xxx.xx

## Issue

- `nvidia-smi` 입력시 평소와 다르게 그래픽카드 정보가 나타나는 현상 발생

## 원인

- 그래픽드라이버 버전 문제로 예상

## 해결

- compatiable 버젼 설치 후, 재부팅

```bash
sudo apt install nvidia-driver-525
sudo reboot
```

- 버젼 swap 후, 올바르게 인식하는지 확인

```
nvidia-smi
```

```markdown
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  Off  | 00000000:27:00.0 Off |                    0 |
| N/A   29C    P0    60W / 400W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-SXM...  Off  | 00000000:2A:00.0 Off |                    0 |
| N/A   26C    P0    62W / 400W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A100-SXM...  Off  | 00000000:51:00.0 Off |                    0 |
| N/A   27C    P0    60W / 400W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A100-SXM...  Off  | 00000000:57:00.0 Off |                    0 |
| N/A   29C    P0    60W / 400W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA A100-SXM...  Off  | 00000000:88:00.0 Off |                    0 |
| N/A   29C    P0    62W / 400W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA A100-SXM...  Off  | 00000000:8E:00.0 Off |                    0 |
| N/A   27C    P0    59W / 400W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   6  NVIDIA A100-SXM...  Off  | 00000000:A5:00.0 Off |                    0 |
| N/A   26C    P0    57W / 400W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   7  NVIDIA A100-SXM...  Off  | 00000000:A8:00.0 Off |                    0 |
| N/A   29C    P0    62W / 400W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
