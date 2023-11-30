---
layout: post
title: torch tensor manipulation2
subtitle: squeeze, unsqueeze
tags: [python, Pytorch]
categories: python
use_math: true
comments: true
published: true
---

## `squeeze`, `unsqueeze` 함수 정리

### squeeze

- size가 `1`인 dimension 삭제

```python
x = torch.ones(10, 5, 1, 1)
x.shape
>>> torch.Size([10, 5, 1, 1])
```

```python
xs1 = x.squeeze()
xs1.shape
>>> torch.Size([10, 5])
```

- size가 1인 차원 **일부** 삭제 가능

```python
xs2 = x.squeeze(dim = 2)
xs2.shape
>>> torch.Size([10, 5, 1])
```

- `-1`값을 넣어 접근가능

```python
x3 = x.squeeze(dim = -1)
x3.shape
>>> torch.Size([10, 5, 1])
```

- size가 1이 아닌 차원 삭제는 불가능

```python
x4 = x.squeeze(dim = 1)
x4.shape
>>> torch.Size([10, 5, 1, 1])
```

- tuple로도 접근이 가능
  - `torch` 2.0 버전부터 사용 가능

```python
x = torch.zeros(2, 1, 3, 1, 8)
print(f"x shape: {x.shape}")
>>> x shape: torch.Size([2, 1, 3, 1, 8])


y = torch.squeeze(x, (1, 2, 3))
print(f"y shape: {y.shape}")
>>> y shape: torch.Size([2, 3, 8])


z = torch.squeeze(x, (1, 3))
print(f"z shape: {z.shape}")
>>> z shape: torch.Size([2, 3, 8])
```

## unsqueeze

- squeeze함수와 반대의 기능
- 지정한 dimension 자리에 **size가 1인 빈 공간을 채워주면서 차원을 확장**

```python
x = torch.ones(3, 5, 7)
x.shape
>>> torch.Size([3, 5, 7])
```

- 0번과 1번 사이에 dimension 추가

```python
x1 = x.unsqueeze(dim = 1)
x1.shape
>>> torch.Size([3, 1, 5, 7])
```

- 마찬가지로 `-1`로 접근 가능

```python
x2 = x.unsqueeze(dim = -1)
x2.shape
torch.Size([3, 5, 7, 1])
```

- 원본 데이터의 차원보다 큰 숫자를 넣는경우엔 오류 발생

```python
x3 = x.unsqueeze(dim = 4)
x3.shape

---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/home/bigster/moef/tmp.ipynb 셀 61 line 2
      1 # 애초에 없는 차원으로 접근할 경우 오류 발생
----> 2 x3 = x.unsqueeze(dim = 4)
      3 x3.shape

IndexError: Dimension out of range (expected to be in range of [-4, 3], but got 4)
```


## Reference

- 
- 
- 