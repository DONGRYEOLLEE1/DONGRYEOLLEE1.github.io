---
layout: post
title: Connect to server in VSCode
subtitle: SSH
tags: [VScode, SSH]
categories: Developing
use_math: true
comments: true
published: true
---

## VScode로 서버에 접속하여 개발

1. 확장 -> `Remote - SSH` 설치
2.  'CTRL + SHIFT + P' 눌러 config 파일을 선택

```
Host [display_name]
    HostName [ip addr]
    User [account_name]
    Port [port number] (default : 22)
```

3. 'CTRL + SHIFT + P' 눌러, Connect to Host
4. 원격탐색기 -> 원격(터널/SSH) -> SSH -> [display_name]