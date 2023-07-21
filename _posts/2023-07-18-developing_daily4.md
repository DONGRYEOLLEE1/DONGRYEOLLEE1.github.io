---
layout: post
title: Daily of Developing [7/21]
subtitle: Git, Ubuntu
tags: [Git, Ubuntu]
categories: Finetuning
use_math: true
comments: true
published: true
---

## Ubuntu

### 파일 이름 변경

- `mv` 명령어로 파일이름 변경 및 위치 변경 모두 가능!

```
$ mv [old_folder] [new_folder]
```

```
$ ls
abc  bbb.txt
$ mv abc xyz
$ ls
xyz  bbb.txt
```

## GIT

### CA 에러

- `server certificate verification failed. CAfile: none CRLfile: none`
- 인증서 신뢰하도록 설정

```
$ export GIT_SSL_NO_VERIFY=1
또는 전역으로 저장하고 싶으면
$ git config --global http.sslverify false
```

### git 접속

```
$ git config --global user.email "aaa@gmail.com"
$ git config --global user.name "aaa"
```

### 패스워드 정보 묻지 않게 하기

- `credential`기능을 통해 로그인 정보를 저장해 두었다가 다시 입력하지 않아도 사용할 수 있게 만들어줌

```
# default 15분
$ git config --global credential.helper cache  

# 시간 직접 설정
$ git config --global credential.helper 'cache --timout=300'
```

### 여러 파일 한번에 추가

```
$ git add [파일명1] [파일명2] [파일명3]
```

### gitignore

- 이미 커밋된 파일을 `.gitignore` 파일에 추가하여 제거하고 싶은 경우

```
$ git rm -r --cached [committed_file or folder]
$ git add .
$ git commit -m "Applying gitignore"
$ git push
```


## ref

- [[Git] Git에서 CA관련 오류 발생 해결 방법](https://itpro.tistory.com/116)
- [git-add참고](https://coding-groot.tistory.com/110)
- [Git, Pull/Push할 때 id password 묻지 않게 하기](https://pinedance.github.io/blog/2019/05/29/Git-Credential)
- [리눅스 / 파일 또는 디렉토리 이름 바꾸는 방법](https://www.manualfactory.net/10910)