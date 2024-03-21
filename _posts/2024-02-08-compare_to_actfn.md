---
layout: post
title: Compare to Activation Function
subtitle: In transformers & torch
tags: [LLM, ActivationFunction]
categories: Developing
use_math: true
comments: true
published: true
---

## case 1

```python

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification

class CustomBaseModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 3, dr_rate: float = 0.2):
        super(CustomBaseModel, self).__init__()

        self.num_classes = num_classes
        self.dr_rate = dr_rate
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.GELU(),
            nn.Dropout(p = self.dr_rate),
            nn.Linear(self.model.config.hidden_size, num_classes)
        )        

    def forward(self, x):
        ...

model = CustomBaseModel(model_name = "beomi/KcELECTRA-base-v2022").to('cuda')

model

```

```
BaseModel(
  (model): ElectraModel(
    (embeddings): ElectraEmbeddings(
        ...
          )
        )
      )
    )
  )
  (classifier): Sequential(
    (0): Linear(in_features=768, out_features=768, bias=True)
    (1): GELU(approximate='none')
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=768, out_features=3, bias=True)
  )
)
```

## case 2

```python
o_model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-v2022", num_labels = 3).to('cuda')

o_model
```

```
ElectraForSequenceClassification(
  (electra): ElectraModel(
    (embeddings): ElectraEmbeddings(
        ...
          )
        )
      )
    )
  )
  (classifier): ElectraClassificationHead(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): GELUActivation()
    (dropout): Dropout(p=0.1, inplace=False)
    (out_proj): Linear(in_features=768, out_features=3, bias=True)
  )
)
```

## Curiosity

- 모델의 classification head를 직접구현하는 중에 위 model의 `(1): GELU(approximate='none')`와 o_model의 `(activation): GELUActivation()`의 차이점이 궁금

## Answer

```python
# line 929 in transformers.model.electra.modeling_electra.py
from .activations import get_activation
...

class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

```

```python
# line 60 in transformers.activations.py 

class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


# line 209 in transformers.activations.py 
ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": PytorchGELUTanh,
    "gelu_accurate": AccurateGELUActivation,
    ...
    }

ACT2FN = ClassInstantier(ACT2CLS)

def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")
```

- `GELUActivation` class 안에 `torch.nn.functional.gelu` 함수를 통해 구현했음을 확인
- 결론: 똑같다