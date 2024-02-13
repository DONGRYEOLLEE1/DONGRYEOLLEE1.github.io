---
layout: post
title: Utilizing Variable Arguments in Python - args and kwargs
subtitle: Usage Example
tags: [python]
categories: python
use_math: true
comments: true
published: true
---


### argument를 직접 지정한 함수 사용

```python
def print_args(args1, args2):
    print(f'args: [ {args1} ]')
    print(f'args: [ {args2} ]')

print_args("apple", "banana")
```

```
>>> arg1: [ apple ]
>>> arg2: [ banana ]
```

### `*args` 사용

- 키워드 되지 않은 가변적인 갯수의 인자들을 함수에서 필요로 할 때 사용
- 함수의 인자를 몇 개 받을지 모르는 경우에 사용하면 매우 유용
- `*args`는 인자를 tuple로 전달
- 만약 정해지지않은 n개의 인자를 받고 싶을 땐, 다음과 같이 `*args`를 사용 할 수 있음

```python
def print_args(*args):
    for idx, arg in enumerate(args):
        print("arg{}: [ {} ]".format(idx+1, arg))

print_args("apple", "banana", "carrot", "dragon fruit")
```

```
>>> arg1: [ apple ]
>>> arg2: [ banana ]
>>> arg3: [ carrot ]
>>> arg4: [ dragon fruit ]
```

```python
def print_args(*args):
    print(f"input args: {args}")
    print(f"args type: {type(args)}")

print_args("apple", "banana", "carrot", "dragon fruit")
```

```
>>> input args : ('apple', 'banana', 'carrot', 'dragon fruit')
>>> args type : <class 'tuple'>
```

### `**kwargs` 사용
- 키워드된 가변적인 갯수의 인자들을 함수에서 필요로 할 때, `*args`와 `**kwrags`의 차이는 keyword, nonkeyword 차이
- `**kwargs`는 함수에서 여러개의 인자 n개를, key-value 형태로 받을 때 사용하며 인자를 dictionary로 전달

```python
def print_kwargs(**kwargs):
    print(f"input kwargs: {kwargs}\n")

    print("first_name:", kwargs['first_name'])
    print("last_name:", kwargs['last_name'])

print_kwrags(first_name = "daniel", last_name = "lee")
```

```
>>> input kwargs : {'first_name': 'daniel', 'last_name': 'lee'}

>>> first_name: daniel
>>> last_name: lee
```

```python
def print_kwargs(**kwargs):
    print("first_name:", kwargs['first_name'])

    if "last_name" in list(kwargs.keys()):
        print("last_name:", kwargs['last_name'])

print_kwargs(first_name = "daniel", last_name = "lee")
```

```
>>> first_name: daniel
>>> last_name: lee
```

### `*args`와 `**kwargs`를 같이 사용하는 방법

```python
def print_args_kwargs(farg, *args, **kwargs):
    print(farg)

    for arg in args:
        print(arg)

    for key, value in kwargs.items():
        print(f"key = {key}, value = {value}")

print_args_kwargs('farg', 'arg1', 'arg2', 'arg3', kwarg1 = 'Hello', kwarg2 = 'World', kwarg3 = 'Python')
```

```
>>> farg
>>> arg1
>>> arg2
>>> arg3
>>> key = kwarg1, value = Hello
>>> key = kwarg2, value = World
>>> key = karg3, value = Python
```