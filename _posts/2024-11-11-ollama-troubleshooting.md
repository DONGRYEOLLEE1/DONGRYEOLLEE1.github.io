---
layout: post
title: Solving Docker Connection Issues with OpenWebUI and Ollama Models
subtitle: TroubleShooting - No model is listed on the UI
tags: [Ollama, OpenWebUI]
categories: Developing
use_math: true
comments: true
published: true
---

## 구현 사항

- Ollama 설치 후, custom directory 설정을 통해 model storage 확보 + docker + OpenWebUI에서 아무 모델이나 Inference 확인

## 에러

### 1. Ollama 설치 후 모델 다운로드 & OpenWebUI 접속

1. `ollama run llama3.2:1b` -> base directory에 모델 다운로드
2. CLI에서 Inference 확인

```docker
docker run -d -p 3000:8080 --gpus all -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:cuda
```

3. OpenWebUI 접속 후, model selection UI에 모델 리스트 표시 X

### 2. conda로 실행하면 잘 인식?

```bash
open-webui serve
```

1. `localhost:3000`로 OpenWebUI 접속하면 모델 리스트 잘 인식하는 상황

### 3. conf 파일 수정 및 `ollama` 재가동시 에러

1. `sudo vi /etc/systemd/system/ollama.service` -> Service 탭에 있는 환경변수 내용 추가

```
[Service]
...
Environment="OLLAMA_MODELS=/custom/ollama/model_path"
```

2. `systemctl daemon-reload` && `systemctl restart ollama`

3. `systemctl status ollama` -> 아래 이미지처럼 process란에 `(code=exited, status=1/FAILURE)` 발생 + 더불어서 PID가 계속해서 업데이트 되고 있는 상황 (뭔가에 막혀서 실행을 계속해서 반복하는 상황)

![err](https://user-images.githubusercontent.com/8552642/120010593-2b72b400-bf92-11eb-83ef-26f19d0abfed.png)

4. 검색 결과, `1`번에서 설정한 custom directory의 권한 설정으로 인한 충돌이 의심 -> 주석 처리

![err2](/img/ollama/error2.png)

5. `systemctl daemon-reload` && `systemctl restart ollama` && `systemctl status ollama`

![err1](/img/ollama/error1.png)


### 4. docker로 OpenWebUI를 실행하면 왜 모델을 인식하지 못할까?

- container가 host를 찾지 못할 수 있어 아래와 같이 명시해줬으나 모델 인식 여전히 못함. ([open-webui-discussions#4228](https://github.com/open-webui/open-webui/discussions/4228))

```docker
docker run -d -p 3000:8080 --gpus all --add-host=host.docker.internal:192.168.0.17 -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:cuda
```

#### 4-1. 원인

- 터미널에서는 Ollama 서버가 잘 작동했지만 OpenWebUI와 통신하지 못하는 상황 발생. `host.docker.internal:11434`로 연결을 시도했으나 Connection refused 에러 발생.
- 아마도 Docker 컨테이너 내의 네트워크 설정 문제 또는 외부 네트워크와의 통신 오류일 가능성 높음.
- Docker 컨테이너가 `127.0.0.1:11434` (로컬 호스트에 있는 Ollama 서버)로 연결하려고 했으나 네트워크 설정 때문에 통신이 차단된 것. Docker 컨테이너 호스트 컴퓨터의 네트워크와 직접적으로 연결되지 않았기에 발생한 네트워크 연결 문제로 보임

#### 4-2. 해결방안 ([open-webui-discussions#4376](https://github.com/open-webui/open-webui/discussions/4376))

- Docker 컨테이너가 호스트 네트워크를 사용하도록 설정함으로써 문제를 해결 
- `--network=host` flag를 사용해 Docker 컨테이너가 호스트 네트워크에 접근할 수 있도록 허용하여 Ollama 서버와 통신이 가능해진 것.

![issue-#4376](/img/ollama/error3.png)

```docker
docker run -d --network=host -v open-webui:/app/backend/data -e OLLAMA_BASE_URL=http://127.0.0.1:11434 --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

![fin](/img/ollama/fin.png)


## Reference

- [https://github.com/open-webui/open-webui/discussions/4228](https://github.com/open-webui/open-webui/discussions/4228)
- [https://github.com/open-webui/open-webui/discussions/4376](https://github.com/open-webui/open-webui/discussions/4376)
- [https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server)
- [https://www.reddit.com/r/ollama/comments/1de017y/open_web_ui_doesnt_find_downloaded_models/](https://www.reddit.com/r/ollama/comments/1de017y/open_web_ui_doesnt_find_downloaded_models/)
- [https://docs.openwebui.com/getting-started/quick-start/](https://docs.openwebui.com/getting-started/quick-start/)
- [https://github.com/ollama/ollama/issues/2701](https://github.com/ollama/ollama/issues/2701)
- [https://bbs.archlinux.org/viewtopic.php?id=292487](https://bbs.archlinux.org/viewtopic.php?id=292487)