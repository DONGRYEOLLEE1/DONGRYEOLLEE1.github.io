---
layout: post
title: 폐쇄망에서의 딥러닝 환경 구축 이슈
subtitle: 
tags: [CUDA, python, DB, Nvidia]
categories: Developing
use_math: true
comments: true
published: true
---

## 폐쇄망에서의 딥러닝 추론 환경 구축
`경제배움e AI서버 설치 작업`건으로 인해 최초 기재부에 방문했을 당시, 확인한 서버 스펙은 아래와 같음.

- os: Red Hat 8.9
- GPU: A100(80GB) * 1ea
- nvidia driver version: 515.xxx.xx
- cuda: 11.2

최초 설계한 서버 스펙과 전혀 다른 상황 + driver 및 cuda 버전이 너무 낮게 설치되어있음. 준비한 설치 파일 모두 Ubuntu 22.04 기준으로 가져갔기에 설치할 수 없는 상황이며, 해당 파일로 된다 해도 호환성 이슈가 발생할 수 있기에 준비해간 내용을 설치할 수 없는 상황. 더불어, 서버 구성도에 AI서버는 **인터넷망**이라 표기되었으나 실제로 가서 ping 테스트를 해보니 폐쇄망인 것을 확인.

리스트화한 작업목록은 다음과 같음

1. nvidia driver 높은 버전으로 재설치
    - `.run` 파일로 로컬에서 설치 가능한 파일로 준비 (570.xxx.xx)
2. CUDA, cuDNN 재설치
    - 마찬가지로 `.run` 파일로 로컬에서 설치 가능한 파일로 준비 (CUDA 11.8 또는 12.1)
    - `.tar` 파일로 설치 준비 (cuDNN 8.9.7)
3. python 설치
    - 3.10.12 설치
4. CUDA 버전에 맞게 각종 python 패키지 구축
    - 테스트 서버에 CUDA `11.8`, `12.1` 버전에 따른 독립되고 분리되어 있는 가상환경을 구축하여 `pytorch` 버전 확인
    - 다음 사이트에서 whl 파일 다운 가능: [https://download.pytorch.org/whl/torch/](https://download.pytorch.org/whl/torch/)
        - `11.8`
            - torch-2.1.2+cu118-cp310-cp310-linux_x86_64.whl
            - torchaudio-2.1.2+cu118-cp310-cp310-linux_x86_64.whl
            - torchvision-0.16.2+cu118-cp310-cp310-linux_x86_64.whl
        - `12.1`
            - torch-2.1.2+cu121-cp310-cp310-linux_x86_64.whl
            - torchaudio-2.1.2+cu121-cp310-cp310-linux_x86_64.whl
            - torchvision-0.16.2+cu121-cp310-cp310-linux_x86_64.whl
5. 사용되는 모든 모델을 로컬로 실행할 수 있게 파일화
    - `easyocr`: 최초 weight 파일을 다운받는 패키지 구조에 따라 `~/.EasyOCR` 폴더에 weight 파일을 직접 이식
    - `whisper`: 기존 `whisper-python` 패키지가 아닌 로컬로 동작 가능한 weight 파일로 실행
    - `KeyBERT`: basellm 뿐만 아니라 `keybert` 패키지 내에서 `sentence-transformer` 패키지를 사용하여 embedding model도 필요하기 때문에 basellm이 될 수 있는 언어모델과 embedding model의 weight파일을 모두 준비

위 작업 테스트를 위해 GPU가 장착된 RedHat OS 환경이 필요했으나 정확히 동일한 환경을 구축할 수 없었기에 다음 2개의 환경에서 테스트 작업을 수행함.

1. RedHat OS + NoGPU
2. Ubuntu OS + RTX 3090 

### NVIDIA-DRIVER
설치 완료

### CUDA, cuDNN
최초 CUDA 로컬 파일을 통해 설치를 진행하였고 space 용량 부족 에러가 발생했으나 CUDA 설치 전용 화면이 전혀 노출이 안되고 에러메세지가 노출되었고, `tmp/...` 임시 폴더를 생성하는 것을 스크립트에서 발견하여 다른 폴더(`/data/...`)로 임시파일을 생성하게 대처하였지만 여전히 같은 에러가 발생하여 다음의 두 가지 가설을 세웠음. 

1. 최초 파일을 wget을 통해 다운로드하는 과정에서 문제가 발생해 파일이 손상되었다.
    - 다음날, 해당 가설을 검증하기 위해 wget을 통해 재차 CUDA 파일을 다운로드 받았고 파일 용량이 상이함을 확인
    - 파일 무결성 검사(`checksum`)를 통해 이전에 받아간 설치파일이 실제로 손상되었음을 확인
2. 진짜 디스크 용량 부족이다.
    - 검증 절차를 따로 진행하지 않았고 대처방법을 다음 두 가지로 지정
        1) 기존 설치되어있는 `cuda-11.2` 버전을 삭제
        2) 다른 폴더에 설치되어있는 불필요한 파일이나 폴더 제거

#### 파일 무결성 검사 방법
1. [WinMD5Free](https://www.winmd5.com/) 다운로드

2. 원하는 파일 browse -> verify 클릭 -> `checksum value` 확인

    ![img1](/img/etc/img1.png)

3. CUDA 파일 `checksum value` 확인

    ![img2](/img/etc/img2.png)

    ![img3](/img/etc/img3.png)

4. 두 `checksum value` 대조


무결성 테스트를 통과한 CUDA 파일로 설치하려했으나 space 용량 부족으로 설치가 안되었고 `df -h` 명령어를 통해 디스크용량을 봤는데... root mount가 고작 39GB로 할당되어있었음. 작업 디렉터리는 별도로 마운트된 디스크였으나 CUDA 설치 과정에서 생성되는 임시파일 때문에 space 용량 부족 에러가 발생. 기존 설치되어있는 `cuda-11.2` 버전을 삭제 후, `cuda-11.8` 설치 완료

### PYTHON
설치 완료

### PYTHON-PKG
설치 완료

### MODEL
- `easyocr`: `~.EasyOCR/` 폴더에 설치 완료
- `whisper`: `openai/whisper-large-v3` 
- `KeyBERT`: LM + embedding model

### 새로운 이슈
소스코드 이식 후, deployment 과정에서 `urllib.request.urlretrieve` 부분에서 요청을 못 받아오는 것을 확인.
부분적인 테스트를 위해 `wget` 옵션을 통해 해당 파라미터로 쓰인 url 정보를 request 하여 아래의 log가 response되었음을 확인.


```bash
vi wget-log
```

```
...
HTTP request sent. awaiting response... 302 FOUND
Location: https://{DOMAIN}/download/nas?attach_id={...} [fllowing]
--2025-02-06 14:16:49-- https://{DOMAIN}/download/nas?attach_id={...}
Resolving {DOMAIN} ({DOMAIN})... failed: Name or sevice not known.
wget: unable to resolve host address {DOMAIN}
```

HTTPS 인증서 문제로 판단. 아래의 명령어를 통해 재차 테스트

```bash
wget --no-check-certificate "https://{DOMAIN}/download/nas?attach_id={...}"
```

올바르게 파일이 다운로드되었고 `--no-check-certificate` 기능을 파이썬 코드에 이식하기 위해 코드 수정을 진행하여 최종 배포를 완료. 수정된 소스코드는 아래와 같음

```python
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
```

`ssl._create_default_https_context = ssl._create_unverified_context` 기능은 HTTPS 인증서를 검증하지 않도록 설정하고 이는 `--no-check-certificate`와 동일한 역할을 함. 인증서 검증을 비활성화하면 보안 위험이 증가할 수 있으나 해당 환경은 폐쇄망이였기에 문제없이 사용할 수 있었음.


## 폐쇄망 환경구축을 하며 느낀점

1. 잘 못 꿰어진 첫 단추

    기획 → 발주 → 설치 (물리적) → OS 설치 등 기본적인 개발 환경 구축 → 도메인에 맞는 소프트웨어 설치

    이번 사업의 경우 위 일련의 과정이 모두 각기 다른 회사 및 외주를 통해 진행되었던 상태이였기에 소통하기에 매우 어려운 상황이였음. GPU 공급 문제로 인해 `발주 → 설치` 과정에서 스케줄이 다소 연기되었던 상황때문인지 최초에 발주넣은 상태로 서버가 출고되지 않았고 그 사실을 전혀 인지하지 못했던 상황이 이번 작업을 더 어렵게 만들지 않았나 생각됨.

    `기획` 단계에서 산출된 `서버구성도` 문서를 통해 `AI서버 != 폐쇄망`이 아님을 확인하였고 그에 맞는 명령어와 설치파일을 준비해갔으나 OS도 달랐고 폐쇄망 환경에 당황. 서울~세종의 출장루트이기에 말 그대로 하루를 날릴 수 밖에 없었음. 실제로 상주하고계신 직원을 통해 폐쇄망이 아님을 여러차례 확인받았으나 이 접근이 잘못되었음을 깨달았음. 특정 명령어(ex> `ping 8.8.8.8`)를 입력하고 response를 직접 받아서 폐쇄망이 아님을 확인했었어야했음.

2. 사용하는 패키지의 소스코드를 이해하자

    `whisper-python`, `easyocr`, `KeyBERT` 패키지를 사용하였고 weight 파일을 api를 통해 다운받고 해당 weight파일을 통해 inference 동작하는 메커니즘을 생각하지 못했음. `transformers`의 경우엔 huggingface-hub에서 모델 캐시파일을 1회 다운받아 `~/.cache/huggingface/hub` 폴더에 저장하여 다음 사용부터는 해당 폴더에서 모델 weight파일을 load하는 메커니즘을 인지하고 있었으나 `easyocr`의 경우엔 그 용량이 적다고 판단, 패키지내에 모델을 직접 이식해 inference 수행하는 메커니즘으로 생각하였음.

    폐쇄망환경에서 `easyocr` 모델을 load하는 코드를 실행하니 http 에러가 발생하는 것을 발견했고, 패키지를 직접 열어보니 윗 문단에서 기술한 형태의 메커니즘으로 동작함을 발견할 수 있었음. 이 후 모든 패키지의 모델을 load하는 부분의 소스코드를 참고하여 사용되는 모든 모델의 weight 파일을 다운받아 성공적으로 설치할 수 있었음.