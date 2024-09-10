---
layout: post
title: python map function
subtitle: 
tags: [python]
categories: python
use_math: true
comments: true
published: true
---


# map 

`map()` 함수는 파이썬에서 반복 가능한 객체의 모든 요소에 대해 지정된 함수를 적용하는 함수. 원본 데이터를 변형하여 새로운 데이터를 생산하는 데 사용. 이 함수는 lazy evaluation을 사용하여 결과를 바로 계산하지 않고 필요할 때만 값을 계산하는 특성을 지니고 있다.

## 구문

```python
map(function, iterable, ...)
```

## 동작 원리

`map` 함수는 반복 가능한 객체의 각 요소를 순서대로 함수에 전달하고, 그 결과를 돌려주는 **이터레이터**를 반환.

## 예시1: 단일 반복 객체 사용

```python
def square(x):
    return x**2

nums = [1, 2, 3, 4, 5]

result = map(square, nums)

print(list(result))

# 출력: [1, 4, 9, 16, 25]
```

## 예시2: 람다 함수 사용

```python
nums = [1, 2, 3, 4, 5]

result = map(lambda x: x**2, nums)

print(list(result))

# 출력: [1, 4, 9, 16, 25]
```

## 예시3: 여러 반복 객체 사용

```python
def add(x, y):
    return x+y

list1 = [1, 2, 3]
list2 = [4, 5, 6]

print(list(map(add, list1, list2)))

# 출력: [5, 7, 9]
```

## 예시4: 중첩된 리스트 반환

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# 각 행에 대해 map을 사용해 제곱 처리
result = map(lambda row: list(map(lambda x: x ** 2, row)), matrix)

print(list(result))

# 출력: [[1, 4, 9], [16, 25, 36], [49, 64, 81]]
```

## 예시5: 조건에 따른 반환

```python
nums = [1, 2, 3, 4, 5, 6]

result = map(lambda x: x**2 if x % 2 == 0 else x, nums)

print(list(result))

# 출력: [1, 4, 3, 16, 5, 36]
```

## 예시6: 복잡한 객체 반환

```python
people = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30},
    {'name': 'Charlie', 'age': 35}
]

# 나이 값을 1씩 증가시키는 map
result = map(lambda person: {**person, 'age': person['age'] + 1}, people)

print(list(result))

# 출력: [{'name': 'Alice', 'age': 26}, {'name': 'Bob', 'age': 31}, {'name': 'Charlie', 'age': 36}]
```

## 장점

- code readability improvement
- memory efficiency improvement

