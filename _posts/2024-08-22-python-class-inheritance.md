---
layout: post
title: python abstract class
subtitle:
tags: [python]
categories: python
use_math: true
comments: true
published: true
---


## Claass - Inheritance

### 상속의 기본 개념

1. 부모 클래스 (Super/Parent Class): 다른 클래스에게 속성과 메서드를 물려주는 클래스입니다.
2. 자식 클래스 (Sub/Child Class): 부모 클래스의 속성과 메서드를 물려받는 클래스입니다.

### Examples

```python
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def work(self):
        print(f"{self.name} is working.")

class Manager(Employee):
    def __init__(self, name, salary, department):
        super().__init__(name, salary)  # 부모 클래스의 초기화 메서드 호출
        self.department = department

    def work(self):
        print(f"{self.name} is managing the {self.department} department.")

class Developer(Employee):
    def __init__(self, name, salary, programming_language):
        super().__init__(name, salary)  # 부모 클래스의 초기화 메서드 호출
        self.programming_language = programming_language

    def work(self):
        print(f"{self.name} is writing code in {self.programming_language}.")
```

### 장점

1. 코드 재사용: `Employee` 클래스의 `name과` `salary` 초기화 부분을 `Manager와` `Developer` 클래스에서 다시 작성할 필요가 없습니다.
2. 유지보수성 향상: 공통적인 기능은 부모 클래스에 한 번만 정의하면 되므로, 이후 변경 사항이 있을 때 부모 클래스만 수정하면 됩니다.
3. 확장성: 자식 클래스는 부모 클래스의 기능을 상속받으면서도 자신만의 고유한 기능을 추가할 수 있습니다.



## Abstract Class

```python
# raptor/Retrievers.py
from abc import ABC, abstractmethod
from typing import List

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> str:
        pass


# raptor/tree_retriever.py
from .Retrievers import BaseRetriever

class TreeRetriever(BaseRetriever):

    def __init__(self, config, tree) -> None:
        ...

    def retrieve(self, query, start_layer, num_layers, top_k, max_tokens, collapse_tree, return_layer_information) -> str:
        ...

```

위 코드에서 `BaseRetriever` 클래스를 상속하는 이유는 추상클래스를 통해 일관된 인터페이스를 정의하고 강제하기 위함. 

- **일관된 인터페이스 제공**: `BaseRetriever` 클래스는 `retrieve`라는 추상 메서드를 정의하고 있습니다. 이 메서드를 상속받는 모든 서브클래스는 반드시 `retrieve` 메서드를 구현해야 합니다. 이 덕분에, 개발자는 `BaseRetriever`를 상속받은 모든 클래스가 동일한 인터페이스(`retrieve`)를 제공할 것이라고 확신할 수 있다.
- **강제적인 구현**: 추상 클래스(`ABC`)의 추상 메서드(`abstractmethod`)는 서브클래스에서 반드시 구현해야 합니다. 즉, `TreeRetriever`가 `BaseRetriever`를 상속받는 이상, `retrieve` 메서드를 구현하지 않으면 오류가 발생합니다. 이렇게 함으로써, 개발자가 특정 기능을 빠뜨리지 않고 구현하도록 강제할 수 있습니다.
- **확장성과 유지보수성**: 나중에 `TreeRetriever` 외에 다른 리트리버(`Retriever`) 클래스들이 생긴다면, 이들도 `BaseRetriever`를 상속받아 `retrieve` 메서드를 구현하게 됩니다. 이렇게 하면, 다양한 리트리버들이 공통된 방식으로 동작하고 교체 가능하게 됩니다. 예를 들어, 동일한 `retrieve` 인터페이스를 사용해 여러 종류의 리트리버를 다룰 수 있는 구조가 됩니다.

### 예시 코드

```python
class TextRetriever(BaseRetriever):
    def retrieve(self, query: str) -> str:
        # 텍스트 기반으로 데이터 검색
        return "Text result"

class ImageRetriever(BaseRetriever):
    def retrieve(self, query: str) -> str:
        # 이미지 기반으로 데이터 검색
        return "Image result"

# 동일한 인터페이스로 다양한 Retriever 사용 가능
def get_result(retriever: BaseRetriever, query: str):
    return retriever.retrieve(query)

text_retriever = TextRetriever()
image_retriever = ImageRetriever()

print(get_result(text_retriever, "example query"))  # "Text result"
print(get_result(image_retriever, "example query"))  # "Image result"
```