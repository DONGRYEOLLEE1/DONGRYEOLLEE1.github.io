---
layout: post
title: 가상환경 이슈
subtitle: 
tags: [python]
categories: Developing
use_math: true
comments: true
published: true
---

# Env

- OS :  Ubuntu 22.04
- python : 3.10.12

## Issue

- 가상환경 설치 후, activate 불가 현상
- `/test/bin/` 폴더에 `activate` 파일이 없음

```
$ python3 -m venv test
>>> Error: Command '['/home/ubuntu/test/bin/python3', '-m', 'ensurepip', '--upgrade', '--default-pip']' returned non-zero exit status 1.
```

```
$ source test/bin/activate
>>> bash: test/bin/activate: No such file or directory
```

## 해결

```
$ sudo apt install python3.10-venv
$ python3 -m venv test
$ source test/bin/activate

>>> (test) user@server:~$ 
```
