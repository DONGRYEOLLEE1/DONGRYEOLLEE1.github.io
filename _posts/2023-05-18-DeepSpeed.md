---
layout: post
title: DeepSpeed Finetuning시 마주한 에러들
subtitle: Trouble-shooting
tags: [Finetuning, DeepSpeed]
categories: NLP
use_math: true
comments: true
---

DeepSpeed를 통한 FT 구동중, 직면한 여러가지 오류에 대해 기술해놓습니다.


# Env
ubuntu 22.04
python3.10.6
cuda 11.8
pytorch 2.0.1+cu118
transformers 4.28.1
accelerate 0.19.0
sentencepiece 0.1.99
tokenizers 0.13.3
ninja 1.11.1
deepspeed 0.9.2


## 1️⃣

```
No module named '_bz2'
```

```python
# Solution
sudo apt-get install libbz2-dev
cd Python-3.10.6
./configure
make
sudo make install
```

sudo 명령어를 통해 패키지 설치 이후 python re-compile 필요!


## 2️⃣

```
NO module named '_lzma'
```

```python
# Solution
sudo apt install liblzma-dev
sudo cp /usr/lib/python3.8/lib-dynload/_bz2.cpython-38-x86_64-linux-gnu.so /usr/local/lib/python3.8/
sudo cp /usr/lib/python3.8/lib-dynload/_lzma.cpython-38-x86_64-linux-gnu.so /usr/local/lib/python3.8/
```

패키지 설치 이후, 해당 경로에 파일 복사 해주면 끝

## 3️⃣

아래 너무 많은 에러로 모두 표시

```
ninja: build stopped: subcommand failed.

subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.

RuntimeError: Error building extension 'fused_adam'

ImportError: /home/test/.cache/torch_extensions/py310_cu118/fused_adam/fused_adam.so: cannot open shared object file: No such file or directory

ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 326830) of binary: /home/test/lora/bin/python3.10

fused_adam/fused_adam.so: cannot open shared object file: No such file or directory
```

- 내 문제는 CUDA 버젼 문제였음 >> 환경변수 설정 안해줬었음...😥
- 또한 스오택에서 확인 결과, 두 가지 원인으로 조사되었음
  - ninja 패키지 설치 문제
    - https://github.com/zhanghang1989/PyTorch-Encoding/issues/167
  - CUDA 버젼 오류 
    - issue 제기한 유저의 cuda 버젼 11.6 -> 11.5 re-version하니까 오류 안나왔다고 함

```python
vi ~/.bashrc
```

```python
# NVIDIA CUDA Toolkit
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
```
```python
source ~/.bashrc
```


## 4️⃣