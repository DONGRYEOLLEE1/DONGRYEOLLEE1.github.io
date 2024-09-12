---
layout: post
title: python enumerate function
subtitle: 
tags: [python]
categories: python
use_math: true
comments: true
published: true
---


# `enumerate`

`enumerate()`함수는 파이썬에서 list나 tuple과 같은 iterable(이터러블) 객체를 순회하면서, 해당요소의 인덱스와 값을 동시에 반환해주는 함수. 보통 for 반복문과 함께 사용.

## 기본 구문 

```python
enumerate(iterable, start = 0)
```

- **iterable**: 리스트, 튜플, 문자열 등 순회 가능한 객체.
- **start**: 인덱스의 시작 값을 지정. 기본값은 0

## 반환값

`enumerate()`는 이터레이터를 반환하며, 각 요소에 대해 `(index, value)` 형식의 튜플을 생성

## 예시

1. 리스트를 enumerate를 순회

    ```python
    fruits = ['apple', 'banana', 'cherry']

    for index, value in enumerate(fruits):
        print(index, value)
    ```

    ```
    0 apple
    1 banana
    2 cherry
    ```

2. 인덱스를 1부터 시작

    ```python
    for index, value in enumerate(fruits, start = 1):
        print(index, value)
    ```

    ```
    1 apple
    2 banana
    3 cherry
    ```

3. 리스트를 enumerate로 변환하여 사용

    - `enumerate()`는 이터레이터를 반환하기에 리스트로 변환할 수도 있음

    ```python
    enumerated_fruits = list(enumerate(fruits))

    print(enumerated_fruits)
    ```

    ```
    [(0, 'apple'), (1, 'banana'), (2, 'cherry')]
    ```

4. 문자열에서 enumerate 사용

    ```python
    for index, char in enumerate("hello"):
        print(index, char)
    ```

    ```
    0 h
    1 e
    2 l
    3 l
    4 o
    ```

5. 여러 리스트를 동시에 순회하며 인덱스 추적

    - `enumerate()`를 사용해 여러 리스트를 동시에 순회하면서 인덱스를 추적할 수 있습니다. 예를들어, 두 개의 리스트를 병렬로 순회하고, 해당 인덱스에 있는 값들을 합산하는 경우

    ```python
    list1 = [10, 20, 30, 40]
    list2 = [1, 2, 3, 4]

    for idx, (a, b) in enumerate(zip(list1, list2)):
        print(f"Index {idx}: {a} + {b} = {a+b}")
    ```

    ```
    Index 0: 10 + 1 = 11
    Index 1: 20 + 2 = 22
    Index 2: 30 + 3 = 33
    Index 3: 40 + 4 = 44
    ```

    -  `zip()` 함수를 사용해 두 리스트의 각 요소를 튜플로 묶고, `enumerate()`로 각 튜플에 인덱스를 부여한 후, 값을 처리

6. 리스트 내 특정 조건을 만족하는 요소의 인덱스 찾기

    - `enumerate()`를 사용하면 리스트에서 특정 조건을 만족하는 요소의 인덱스를 쉽게 찾을 수 있습니다.

    ```python
    numbers = [10, 25, 30, 45, 50, 65]

    # 50보다 큰 수들의 인덱스와 값을 출력
    for idx, num in enumerate(numbers):
        if num > 50:
            print(f"Index {idx}: {num}")
    ```

    ```
    Index 5: 65
    ```

7. 리스트 수정 시 enumerate 사용하기

    - 리스트의 값을 수정할 때도 `enumerate()`를 사용하여 인덱스를 손쉽게 다룰 수 있습니다.

    ```python
    numbers = [1, 2, 3, 4, 5]

    # 각 요소에 인덱스를 곱해서 리스트를 수정
    for idx, num in enumerate(numbers):
        numbers[idx] = idx * num

    print(numbers)
    ```

    ```
    [0, 2, 6, 12, 20]
    ```

8. 중첩된 리스트에스 enumerate 사용하기

    - 중첩된 리스트(2차원)에서 `enumerate()`를 사용하여 인덱스를 추적하는 예제

    ```python
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    # 2차원 리스트의 행과 열 인덱스를 함께 출력
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            print(f"Row {row_idx}, Col {col_idx}: {value}")
    ```

    ```
    Row 0, Col 0: 1
    Row 0, Col 1: 2
    Row 0, Col 2: 3
    Row 1, Col 0: 4
    Row 1, Col 1: 5
    Row 1, Col 2: 6
    Row 2, Col 0: 7
    Row 2, Col 1: 8
    Row 2, Col 2: 9
    ```

9. List comprehension에서 enumerate 사용하기

    - `enumerate()`는 list comprehension과 결합해 더 간결합 코드를 작성할 수 있습니다.

    ```python
    words = ['apple', 'banana', 'cherry']

    indexed_words = [(i, word.upper()) for i, word in enumerate(words)]

    print(indexed_words)
    ```

    ```
    [(0, 'APPLE'), (1, 'BANANA'), (2, 'CHERRY')]
    ```

10. 로그 파일에서 오류 라인 찾기

    - 개발 중에 로그 파일을 분석하면서, 특정 키워드가 포함된 라인을 추적하고 싶을 때 `enumerate()`를 사용하면 유용합니다. 예를 들어, "ERROR"라는 단어가 포함된 라인의 인덱스를 찾는 예제입니다.

    ```python
    log_lines = [
        "INFO: System started.",
        "WARNING: Low memory.",
        "ERROR: Failed to load configuration.",
        "INFO: System running.",
        "ERROR: Disk space low."
    ]

    for line_num, line in enumerate(log_lines, start = 1):
        if "ERROR" in line:
            print(f"Error found at line {line_num}: {line}")
    ```

    ```
    Error found at line 3: ERROR: Failed to load configuration.
    Error found at line 5: ERROR: Disk space low.
    ```

11. 데이터 분석에서의 열 인덱스 추적

    - CSV파일 과 같은 표 형식의 데이터를 다룰 때, 특정 열의 인덱스를 추적하여 데이터를 처리 할 수 있습니다. 

    ```python
    data = [
        ["Name", "Age", "City"],
        ["Alice", "30", "New York"],
        ["Bob", "25", "Los Angeles"],
        ["Charlie", "35", "Chicago"]
    ]

    # 열 인덱스 찾기 (City열을 추출)
    header = data[0]

    for col_idx, col_name in enumerate(header):
        if col_name == "City":
            city_idx = col_idx

    # "City"열의 데이터를 추출
    cities = [row[city_idx] for row in data[1:]]
    print(cities)
    ```

    ```
    ['New York', 'Los Angeles', 'Chicago']
    ```

12. 코드 가독성 향상
    
    ```python
    # Before
    items = ['apple', 'banana', 'cherry']
    index = 0

    for item in items:
        print(f"Item {index}: {item}")
        index += 1
    ```

    ```python
    # After
    items = ['apple', 'banana', 'cherry']
    for index, item in enumerate(items):
        print(f"Item {index}: {item}")
    ```

## 장점

- 불필요한 변수 관리를 없애고 인덱스와 데이터를 함께 활용해 코드 가독성을 향상
- 리스트 뿐만 아니라 문자열, 튜플 등 모든 이터러블 객체에 사용할 수 있음