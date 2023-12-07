---
layout: post
title: Change model input sequence length
subtitle: Example - Electra model type
tags: [NLP, Finetuning, LLM]
categories: Developing
use_math: true
comments: true
published: true
---

## Env

- OS: Ubuntu 22.04
- python: 3.10.12
- transformers: 4.31.0.dev0

## Model Input Sequence Length

- `transformers` 라이브러리 사용하고 Input Sequence 길이를 더 길게 변경해주고 싶을때
- 물론 `tokenizer`의 `padding = "max_length"`, `truncation = True`, `max_length = 2048`와 같은 옵션을 줘서 문장 길이를 맞춰줄 수도 있지만 backbone 모델의 input sequence length가 맞추려는 길이보다 짧을 경우 다음과 같은 에러 발생

```python
>>> model(tokenizer(text, return_tensors = 'pt')['input_ids'].to('cuda'))

---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-92-4ef402b4a33e> in <cell line: 1>()
----> 1 model(tokenizer(text, return_tensors = 'pt')['input_ids'].to('cuda'))

5 frames
/usr/local/lib/python3.10/dist-packages/transformers/models/electra/modeling_electra.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states, return_dict)
    878             if hasattr(self.embeddings, "token_type_ids"):
    879                 buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
--> 880                 buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    881                 token_type_ids = buffered_token_type_ids_expanded
    882             else:

RuntimeError: The expanded size of the tensor (554) must match the existing size (512) at non-singleton dimension 1.  Target sizes: [1, 554].  Tensor sizes: [1, 512]
```

- 이런 case의 경우, model의 input 길이를 동적으로 변경해줘야 사용 가능

```python
model_name = "beomi/KcELECTRA-base-v2022"
new_max_len = 2048

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_length = new_max_len
config = ElectraConfig.from_pretrained(model_name)
config.max_position_embeddings = new_max_len

model = ElectraForSequenceClassification.from_pretrained(model_name, config = config, ignore_mismatched_sizes = True).to('cuda')
```

```python
>>> print(model(tokenizer(text, return_tensors = 'pt')['input_ids'].to('cuda')))
>>> print(len(tokenizer(text, return_tensors = 'pt')['input_ids'].to('cuda')[0]))

SequenceClassifierOutput(loss=None, logits=tensor([[-0.0133,  0.1915]], device='cuda:0', grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
692
```