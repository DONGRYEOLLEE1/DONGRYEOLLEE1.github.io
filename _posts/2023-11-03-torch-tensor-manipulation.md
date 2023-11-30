---
layout: post
title: torch tensor manipulation1
subtitle: view, reshape, transpose
tags: [python, Pytorch]
categories: python
use_math: true
comments: true
published: true
---

## `view`, `reshape`, `transpose` 함수 정리

```python
t = torch.tensor([[[0, 1], [2, 3], [4, 5]], \
                 [[6, 7], [8, 9], [10, 11]], \
                 [[12, 13], [14, 15], [16, 17]], \
                 [[18, 19], [20, 21], [22, 23]]])
print(t)
print(t.size())
```
```
tensor([[[ 0,  1],
         [ 2,  3],
         [ 4,  5]],

        [[ 6,  7],
         [ 8,  9],
         [10, 11]],

        [[12, 13],
         [14, 15],
         [16, 17]],

        [[18, 19],
         [20, 21],
         [22, 23]]])
torch.Size([4, 3, 2])
```

### torch.view

- `view` 함수를 통해 만들어지는 tensor는 contiguous하다.
  - `tv[0][0][0] -> tv[0][0][1] -> tv[0][0][2]` == `0 -> 1 -> 2`

- 복사 없이 `torch.shape`을 통해 `self` tensor를 확인할 경우 `contiguous()`가 불분명할 수도 있음. `view()`를 통해선 불분명하니 `reshape()`함수를 사용할 것을 권장
>When it is unclear whether a view() can be performed, it is advisable to use reshape(), which returns a view if the shapes are compatible, and copies (equivalent to calling contiguous()) otherwise.

```python
tv = t.view(4, 2, 3)
print(tv)
print(tv.size())
print(tv.is_contiguous())
```

```
tensor([[[ 0,  1,  2],
         [ 3,  4,  5]],

        [[ 6,  7,  8],
         [ 9, 10, 11]],

        [[12, 13, 14],
         [15, 16, 17]],

        [[18, 19, 20],
         [21, 22, 23]]])
torch.Size([4, 2, 3])
True
```

- 데이터의 index 순서대로 flatten시켜주는 함수를 통해 `t`와 `tv`를 비교했을떄 정확히 일치

```python
t.flatten() == tv.flatten()
>>> tensor([True, True, True, True, True, True, True, True, True, True, True, True,
>>>         True, True, True, True, True, True, True, True, True, True, True, True])
```

- 또한 `t`와 `tv`의 데이터는 pointer값이 동일하여 한 쪽의 data를 수정하면 다른 쪽의 값도 동시에 변경

```python
t[0][0][0] = 99
tv[0][0][0]
>>> tensor(99)
```

- 중간에 `-1`을 넣으면 해당 위치에 새로운 차원이 하나 더 생김

```python
print(t.view(4, 3, -1, 2).shape)
print(t.view(4, 3, -1, 2))
```

```
torch.Size([4, 3, 1, 2])
tensor([[[[99,  1]],

         [[ 2,  3]],

         [[ 4,  5]]],


        [[[ 6,  7]],

         [[ 8,  9]],

         [[10, 11]]],


        [[[12, 13]],

         [[14, 15]],

         [[16, 17]]],


        [[[18, 19]],

         [[20, 21]],

         [[22, 23]]]])
```


### torch.transpose

- 보통 `(batch_size, hidden_state, input_dim)`을 `(batch_size, input_dim, hidden_state)`으로 변경해주는 작업을 할때 사용 (Dimension Swap)
> Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
- If input is a sparse tensor then the resulting out tensor does not share the underlying storage with the input tensor.
- Parameters
  - input (Tensor) – the input tensor.
  - dim0 (int) – the first dimension to be transposed
  - dim1 (int) – the second dimension to be transposed


```python
tt = t.transpose(2, 1)
print(t.size())
print(t)
print(tt.size())
print(tt)
```

```
torch.Size([4, 3, 2])
tensor([[[99,  1],
         [ 2,  3],
         [ 4,  5]],

        [[ 6,  7],
         [ 8,  9],
         [10, 11]],

        [[12, 13],
         [14, 15],
         [16, 17]],

        [[18, 19],
         [20, 21],
         [22, 23]]])
torch.Size([4, 2, 3])
tensor([[[99,  2,  4],
         [ 1,  3,  5]],

        [[ 6,  8, 10],
         [ 7,  9, 11]],

        [[12, 14, 16],
         [13, 15, 17]],

        [[18, 20, 22],
         [19, 21, 23]]])
```

- 위 결과와 같이, `transpose`함수는 **contiguous하지 않음**
- 즉, 물리적 순서가 다르다
- `tt[0][0][0] -> tt[0][0][1] -> tt[0][0][2]` == `99 -> 2 -> 4`

```python
tt.is_contiguous()
>>> False
```

```python
print(tt[0][0][0], tt[0][0][1], tt[0][0][2])
>>> tensor(99) tensor(2) tensor(4)
```

- 당연히 `flatten()`했을때 결과도 다르다

```python
tt.flatten() == t.flatten()
>>> tensor([ True, False, False, False, False,  True,  True, False, False, False,
>>>         False,  True,  True, False, False, False, False,  True,  True, False,
>>>         False, False, False,  True])
```

- 또한 데이터 포인터를 공유하지 않는다

```python
tt.contiguous().storage().data_ptr() == tt.storage().data_ptr()
>>> False
```

### torch.reshape

- Parameters
  - input (Tensor) - the tensor to be reshaped
  - shape (tuple or int) - the new shape
- `reshape()` == `contiguous().view()`와 같은 개념
- view는 contiguous하지 않은 tensor에 대해서는 적용할 수 없음

```python
tt.view(4, 3, 2)

---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/home/bigster/moef/tmp.ipynb 셀 26 line 1
----> 1 tt.view(4, 3, 2)

RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

- `reshape`함수는 직관적으로 dimension을 변경시켜 줄 수 있음

```python
tt.contiguous().view(4, 3, 2)
```

```
tensor([[[99,  2],
         [ 4,  1],
         [ 3,  5]],

        [[ 6,  8],
         [10,  7],
         [ 9, 11]],

        [[12, 14],
         [16, 13],
         [15, 17]],

        [[18, 20],
         [22, 19],
         [21, 23]]])
```

- `reshape`은 강제로 tensor를 contiguous하게 만들면서 shape을 변경 가능하게 만들어줌

```python
tt.reshape(4, 3, 2)
```

```
tensor([[[99,  2],
         [ 4,  1],
         [ 3,  5]],

        [[ 6,  8],
         [10,  7],
         [ 9, 11]],

        [[12, 14],
         [16, 13],
         [15, 17]],

        [[18, 20],
         [22, 19],
         [21, 23]]])
```

```python
tt.reshape(4, 3, 2).is_contiguous()
>>> True
```

- `shape` 파라미터에 tuple형태로 입력 가능
- 당연한 얘기겠지만 tuple(=shape)값은 input tensor의 `W x H x B` 값의 Factor(인수)값을 조합해 입력해줘야함

```python
a = torch.arange(4.)

print(a, a.size())
>>> tensor([0., 1., 2., 3.]) torch.Size([4])
```

```python
ar = torch.reshape(a, (4, 1))
print(ar, ar.size())
>>> tensor([[0.],
>>>         [1.],
>>>         [2.],
>>>         [3.]]) torch.Size([4, 1])
```

- `(-1,)` tuple형태로 생긴 `-1` 인자를 넣어주면 `flatten()`기능. 동시에 contiguous함
> A single dimension may be -1, in which case it’s inferred from the remaining dimensions and the number of elements in input.

```python
b = torch.tensor([[0, 1], [2, 3], [4, 5]])
print(b.size())
>>> torch.Size([3, 2])
```

```python
br = torch.reshape(b, (-1,))
print(br, br.size())
>>> tensor([0, 1, 2, 3, 4, 5]) torch.Size([6])

torch.reshape(b, (-1,)) == b.flatten()
>>> tensor([True, True, True, True, True, True])

torch.reshape(b, (-1,)).is_contiguous()
>>> True
```


## 결론
- `view` : tensor에 저장된 데이터의 물리적 위치 순서와 index순서가 일치할 때 contiguous shape을 재구성한다. 때문에 항상 contiguous하다는 성질이 보유
- `transpose` : tensor에 저장된 데이터의 물리적 위치 순서와 상관없이 수학적 의미의 transpose를 수행. 물리적 위치와 transpose가 수행된 tensor의 index순서는 같다는 보장이 없으므로 항상 contiguous하지 않음
- `reshape` : tensor에 저장된 데이터의 물리적 위치 순서와 index순서가 일치하지 않도록 shape을 재구성한 이후에 강제로 일치시킴. 때문에 항상 contiguous하다는 성질이 보유


## Reference

- [pytorch-documentation](https://pytorch.org/docs/stable/tensors.html)
- [PyTorch의 view, transpose, reshape 함수의 차이점 이해하기](https://inmoonlight.github.io/2021/03/03/PyTorch-view-transpose-reshape/)
- [파이토치 view 텐서 차원 변경 (torch.view, shape 변경)](https://noanomal.tistory.com/entry/%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98-view-%ED%85%90%EC%84%9C-%EC%B0%A8%EC%9B%90-%EB%B3%80%EA%B2%BD-torchview-shape-%EB%B3%80%EA%B2%BD)