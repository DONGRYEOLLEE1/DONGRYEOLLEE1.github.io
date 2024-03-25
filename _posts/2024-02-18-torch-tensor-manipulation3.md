---
layout: post
title: torch tensor manipulation3
subtitle: 
tags: [python, Pytorch]
categories: python
use_math: true
comments: true
published: true
---

## broadcasting


### indexing & slicing

```python
import torch

x = torch.arange(0, 256, dtype = torch.int64, device = 'cpu').float()
print(f"original x size: {x.size()}")
x_expanded = x[None, :, None, None]
print(f"expanded x size: {x_expanded.size()}")
```

```python
original x size: torch.size([256])
expanded x size: torch.Size([1, 256, 1, 1])
```

```python
x = torch.randn(4, 4, 7, 3)
print(x.size())

x_ = x[None, :, None, :, None, :, None, :]
print(x_.size())
```

```python
torch.Size([4, 4, 7, 3])
torch.Size([1, 4, 1, 4, 1, 7, 1, 3])
```

- `None` 지정해주는만큼 차원 확장가능하나 전체 인덱싱(:)은 원래 텐서 갯수만큼 할당 가능

### expand

```python
x = 10000 ** (torch.arange(0, 256, dtype = torch.int64, device = 'cpu').float() / 256)
x = x[None, :, None].float()

print(f"before slicing x shape: {x.size()}")
```

```python
torch.Size([1, 256, 1])
```

```python
x_expanded = x.expand(100, -1, 50)
print(f"before slicing x_expanded shape: {x_expanded.size()}")
```

```python
torch.Size([100, 256, 50])
```

- 특정 차원의 텐서를 반복해서 생성해주는 함수이며 특정 차원의 크기가 1인 tensor에만 적용 가능
- `expand(-1)`: 차원유지
- `expand(n)`: 차원의 크기가 1인 tensor에 n만큼 차원 확장

### triu & tril

#### 삼각행렬, triangular matrix

- 선형대수에서 삼각행렬은 정사각행렬의 특수한 경우로 주대각선을 기준으로 대각항의 위쪽이나 아래쪽 항들의 값이 모두 0인 경우를 의미

![삼각행렬1](https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Diagonal-matrix001.svg/200px-Diagonal-matrix001.svg.png)

- 하삼각행렬(lower triangular matrix)

![하삼각행렬](https://wikimedia.org/api/rest_v1/media/math/render/svg/2ac0792d868a00d4ecf60a3d944ac51b3eaf4c2f)

- 상삼각행렬(upper triangular matrix)

![상삼각행렬](https://wikimedia.org/api/rest_v1/media/math/render/svg/69ff34f2380e989eb19c29e457e425dbfbc0f99c)

#### triu

- 상삼각행렬 만들어주는 함수

```python
in_tensor = torch.randn(3, 3)
print(in_tensor)
```

```python
tensor([[ 1.1567,  1.6308, -2.5685],
        [-0.9405, -0.3915, -1.8426],
        [-0.2034,  0.2296,  0.5827]])
```

```python
upper_tri_matrix = torch.triu(in_tensor, diagonal = 0)
print(f"Upper triangular Matrix:\n{upper_tri_matrix}")
```

```python
Upper triangular Matrix:
tensor([[ 1.1567,  1.6308, -2.5685],
        [ 0.0000, -0.3915, -1.8426],
        [ 0.0000,  0.0000,  0.5827]])
```

#### tril 

- 하삼각행렬 만들어주는 함수

```python
lower_tri_matrix = torch.tril(in_tensor)
print(f"Lower triangular Matrix:\n{lower_tri_matrix}")
```

```python
Lower triangular Matrix:
tensor([[ 1.1567,  0.0000,  0.0000],
        [-0.9405, -0.3915,  0.0000],
        [-0.2034,  0.2296,  0.5827]])
```