---
layout: post
title: python enum
subtitle: 
tags: [python]
categories: python
use_math: true
comments: true
published: true
---


# `enum`

- `enum`(열거형)은 이름이 있는 상수들의 집합을 정의하는 데 사용됩니다. 보통 특정 값들 중 하나만 선택할 수 있는 상황에서 유용합니다. 예를 들어, 요일이나 색상, 상태와 같은 값을 나타낼 때 사용할 수 있습니다. `enum`을 사용하면 가독성이 높아지고, 코드에서 잘못된 값이 사용되는 것을 방지할 수 있습니다.

## 특징

1. 고유한 값: 열거형의 각 멤버는 고유한 값이어야하며, 이름도 중복될 수 없습니다.
2. 순회 가능: 열거형 클래스의 멤버들을 for루프에서 순회할 수 있습니다.
3. 비교 가능: 열거형의 값은 `==`, `is` 등으로 비교할 수 있습니다.
4. 이름과 값에 접근 가능: `.name`으로 멤버의 이름에 접근할 수 있고, `.value`로 상수 값에 접근할 수 있습니다.

## Enum을 사용하는 이유

- 가독성: 숫자나 문자열 상수 대신 의미 있는 이름을 사용하여 가독성을 높입니다.
- 안전성: 실수로 잘못된 값을 사용하는 것을 방지
- 변경 용이성: 상수 값을 변경할 때, 열거형을 사용하면 코드 전체에서 일관성을 유지할 수 있습니다.


## 예제

### 1. 기본 사용법

```python
from enum import Enum

class Color(Enum):
    red = 1
    green = 2
    blue = 3

# Enum 멤버에 접근
print(Color.red)        # Color.red
print(Color.green.name) # 'green'
print(Color.blue.value) # 3
```

### 2. 상태를 나타나는 예제

```python
class Status(Enum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

def check_status(status: Status):
    if status == Status.NEW:
        print("작업이 새로 시작되었습니다.")
    elif status == Status.IN_PROGRESS:
        print("작업이 진행 중입니다.")
    elif status == Status.COMPLETED:
        print("작업이 완료되었습니다.")
    elif status == Status.FAILED:
        print("작업이 실패했습니다.")
    else:
        print("알 수 없는 상태입니다.")

check_status(Status.IN_PROGRESS)    # 작업이 진행 중입니다.
```

### 3. `FastAPI`에서 파라미터값 강제하기

```python
from fastapi import FastAPI
from enum import Enum, unique

app = FastAPI()

@unique
class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"
    lenet_tmp = "lenet"     # error - guaranteed unique

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):

    response = {"model_name": model_name}

    if model_name == ModelName.alexnet:
        response["message"] = "Deep Learning FTW!"
    elif model_name == ModelName.lenet:
        response["message"] = "LeCNN all the images"
    else:
        response["message"] = "Have some residulas"
    
    return response
```