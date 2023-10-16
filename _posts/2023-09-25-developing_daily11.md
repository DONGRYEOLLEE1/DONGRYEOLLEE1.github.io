---
layout: post
title: netplan - No module named 'netifaces'
subtitle: 
tags: [netplan]
categories: Developing
use_math: true
comments: true
published: true
---

# Env

- OS :  Ubuntu 22.04
- python : 3.10.12

## Issue

- 폐쇄망에서 `netifaces` 모듈 오류

```
$ netplan apply
```

```
  File "/usr/sbin/netplan", line 20, in <module>
    from netplan import Netplan
  File "/usr/share/netplan/netplan/__init__.py", line 18, in <module>
    from netplan.cli.core import Netplan
  File "/usr/share/netplan/netplan/cli/core.py", line 24, in <module>
    import netplan.cli.utils as utils
  File "/usr/share/netplan/netplan/cli/utils.py", line 25, in <module>
    import netifaces
ModuleNotFoundError: No module named 'netifaces'
```


## 해결

- ubuntu 기본 파이썬 버전으로 다시 환경 세팅

```
$ python3 -V
>>> Python3.10.12

$ cd /usr/bin/
$ cp python3 python3_backup         (기존 python3 backup)
$ rm python3
$ ln -sf python3.8 python3
$ python3 -V
>>> python3.8.xx

$ netplan apply
```

## Reference

- [ModuleNotFoundError: No module named ‘netifaces‘](https://blog.csdn.net/watt1208/article/details/127391701)