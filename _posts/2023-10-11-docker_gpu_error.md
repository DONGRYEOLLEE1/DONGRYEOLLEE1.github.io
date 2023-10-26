---
layout: post
title: docker - error response from daemon could not select device driver "" with capabilities [[gpu]]
subtitle: 
tags: [Docker]
categories: Developing
use_math: true
comments: true
published: true
---

# Env

- OS :  Ubuntu 20.04

# Issue

```
$ docker run ~~~~~~
docker: error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```

# Solution

- `nvidia-container-toolkit` 설치 

```
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

- docker restart

```
$ sudo systemctl restart docker
```