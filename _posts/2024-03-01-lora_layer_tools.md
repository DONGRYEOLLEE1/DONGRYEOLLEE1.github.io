---
layout: post
title: PEFT new features
subtitle: 
tags: [LLM, LoRA, NLP, peft]
categories: NLP
use_math: true
comments: true
published: true
---

## PEFT v0.9.0

v0.9.0: Merging LoRA weights, new quantization options, DoRA support, and more

### Merging LoRA weights

> With PR #1364, we added new methods for merging LoRA weights together. This is not about merging LoRA weights into the base model. Instead, this is about merging the weights from different LoRA adapters into a single adapter by calling add_weighted_adapter. This allows you to combine the strength from multiple LoRA adapters into a single adapter, while being faster than activating each of these adapters individually.

> Although this feature has already existed in PEFT for some time, we have added new merging methods that promise much better results. The first is based on TIES, the second on DARE and a new one inspired by both called Magnitude Prune. If you haven't tried these new methods, or haven't touched the LoRA weight merging feature at all, you can find more information here:

- 기존 베이스 모델에서 lora weight를 병합하는게 아닌 **로라 모델끼리** 병합하는 것을 의미
- `add_weighted_adapter`를 통해 기능을 지원하며 다수의 adapter를 단일의 adapter로 병합할 수 있는 강력한 기능
- 즉, 각기 다른 task를 가진 lora 모델을 단일의 모델로 병합 가능함을 의미
- combination type
    - cat
    - linear
    - svd
    - ties, ties_svd
    - dare_linear, dare_ties, dare_linear_svd, dare_ties_svd
    - magnitude_prune , magnitude_prune_svd

```python
# peft/examples/multi_adapter_examples/Lora_Merging.ipynb

peft_model_id = "smangrul/tinyllama_lora_norobots"
onfig = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, peft_model_id, adapter_name="norobots")
_ = model.load_adapter("smangrul/tinyllama_lora_sql", adapter_name="sql")
_ = model.load_adapter("smangrul/tinyllama_lora_adcopy", adapter_name="adcopy")

adapters = ["norobots", "adcopy", "sql"]
weights = [2.0, 0.3, 0.7]
adapter_name = "merge"
density = 0.2
combination_type = "ties"

if adapter_name in model.peft_config:
    model.delete_adapter(adapter_name)
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type=combination_type, density=density)

messages = [
    {"role": "user", "content": "Write an essay about Generative AI."},
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")  # , add_special_tokens=False)
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    top_p=0.95,
    temperature=0.2,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0]))
```

```
<s><|im_start|>user 
Write an essay about Generative AI.<|im_end|> 
<|im_start|>assistant 
Generative Artificial Intelligence (GAI) is a type of artificial intelligence that uses machine learning to create art, music and other creations. It's like having a human artist who creates something new without the need for inspiration or motivation.<|im_end|>
```

```python
messages = [
    {"role": "system", "content": "Create a text ad given the following product and description."},
    {
        "role": "user",
        "content": "Product: Sony PS5 PlayStation Console\nDescription: The PS5™ console unleashes new gaming possibilities that you never anticipated.",
    },
]
...
print(tokenizer.decode(outputs[0]))
```

```
<s><|im_start|>system 
Create a text ad given the following product and description.<|im_end|> 
<|im_start|>user 
Product: Sony PS5 PlayStation Console
Description: The PS5™ console unleashes new gaming possibilities that you never anticipated.<|im_end|> 
<|im_start|>assistant 
Ad Text: Experience the next-gen power of the all-new Sony PS5 with its stunning visuals, innovative gameplay features, and more! Get ready to play in style as you experience the future of gaming on your own terms.<|im_end|>
```



```python
text = """Table: 2-11365528-2
Columns: ['Team', 'Head Coach', 'President', 'Home Ground', 'Location']
Natural Query: Who is the Head Coach of the team whose President is Mario Volarevic?
SQL Query:"""

inputs = tokenizer(text, return_tensors="pt")  # , add_special_tokens=False)
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model.generate(
    **inputs, max_new_tokens=64, repetition_penalty=1.1, eos_token_id=tokenizer("</s>").input_ids[-1]
)
print(tokenizer.decode(outputs[0]))
```

```
<s> Table: 2-11365528-2
Columns: ['Team', 'Head Coach', 'President', 'Home Ground', 'Location']
Natural Query: Who is the Head Coach of the team whose President is Mario Volarevic?
SQL Query: SELECT Head Coach FROM 2-11365528-2 WHERE President = Mario Volarevic</s>
```

## ref

- [peft-0.9.0](https://github.com/huggingface/peft/releases/tag/v0.9.0)
- [🤗 PEFT welcomes new merging methods](https://huggingface.co/blog/peft_merging)