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

### triu

