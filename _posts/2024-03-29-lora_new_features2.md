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

## PEFT v0.10.0

v0.10.0: Fine-tune larger QLoRA models with DeepSpeed and FSDP, layer replication, enhance DoRA

### Deprecations

- `prepare_model_for_int8_training` 사용 불가 -> `prepare_model_for_kbit_training` 대체

### Improving DoRA

> Last release, we added the option to enable DoRA in PEFT by simply adding use_dora=True to your LoraConfig. However, this only worked for non-quantized linear layers. With this PEFT release, we now also support Conv2d layers, as well as linear layers quantized with bitsandbytes.

- quantized layer에서도 적용 가능하며 Convolution layer도 지원

### layer replication

> First time contributor @siddartha-RE added support for layer replication with LoRA. This allows you to duplicate layers of a model and apply LoRA adapters to them. Since the base weights are shared, this costs only very little extra memory, but can lead to a nice improvement of model performance. Find out more in our docs.

- mergekit, Mixture of Experts(MOE), SOLAR의 Depth up scaling(DUS) 기능과 함께 모델 파라미터(레이어) expansion 기능

#### Memory efficient Layer Replication with LoRA

> An approach used to improve the performance of models is to expand a model by duplicating layers in the model to build a larger model from a pretrained model of a given size. For example increasing a 7B model to a 10B model as described in the SOLAR paper. PEFT LoRA supports this kind of expansion in a memory efficient manner that supports further fine-tuning using LoRA adapters attached to the layers post replication of the layers. The replicated layers do not take additional memory as they share the underlying weights so the only additional memory required is the memory for the adapter weights. To use this feature you would create a config with the layer_replication argument.

```python
config = LoraConfig(layer_replication=[[0,4], [2,5]], ...)
```

> Assuming the original model had 5 layers [0, 1, 2 ,3, 4], this would create a model with 7 layers arranged as [0, 1, 2, 3, 2, 3, 4]. This follows the mergekit pass through merge convention where sequences of layers specified as start inclusive and end exclusive tuples are stacked to build the final model. Each layer in the final model gets its own distinct set of LoRA adpaters.

> Fewshot-Metamath-OrcaVicuna-Mistral-10B is an example of a model trained using this method on Mistral-7B expanded to 10B. The (adapter_config.json)[https://huggingface.co/abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B/blob/main/adapter_config.json] shows a sample LoRA adapter config applying this method for fine-tuning.

- layer merge 개념과 비슷하나 메모리 효율적인 adapter replication 개념
- layer 단순 반복개념이라 mergekit의 merge method(interpolation)와는 조금 다름

```python
model_id = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit = True, device_map = 'auto')

origin_param = sum(p.numel() for p in model.parameters())
# 4bit: 3,752,071,168 / 8bit: 7,241,732,096

print(len(model.model.layers))    # (0-31) 32 layers

model
```

```
MistralForCausalLM(
  (model): MistralModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x MistralDecoderLayer(
        (self_attn): MistralSdpaAttention(
          (q_proj): Linear8bitLt(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear8bitLt(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear8bitLt(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear8bitLt(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): MistralRotaryEmbedding()
        )
        (mlp): MistralMLP(
          (gate_proj): Linear8bitLt(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear8bitLt(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear8bitLt(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): MistralRMSNorm()
        (post_attention_layernorm): MistralRMSNorm()
      )
    )
    (norm): MistralRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
```

```python
config = LoraConfig(
    r = 8,
    target_modules = ['gate_proj', 'k_proj', 'q_proj', 'v_proj', 'up_proj', 'down_proj', 'o_proj'],
    lora_alpha = 16,
    lora_dropout = 0.05,
    layer_replication = [[0, 16], [8, 24], [16, 32], [16, 32]]    # 16 + 16 + 16 + 16
)
config
```

```
LoraConfig(peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path=None, revision=None, task_type=None, inference_mode=False, r=8, target_modules={'o_proj', 'v_proj', 'k_proj', 'up_proj', 'gate_proj', 'q_proj', 'down_proj'}, lora_alpha=16, lora_dropout=0.05, fan_in_fan_out=False, bias='none', use_rslora=False, modules_to_save=None, init_lora_weights=True, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={}, megatron_config=None, megatron_core='megatron.core', loftq_config={}, use_dora=False, layer_replication=[[0, 16], [8, 24], [16, 32], [16, 32]])
```

- layer_replication:
    - [0, 16]: [0, 16) -> 16개 레이어 
    - [8, 24]: [8, 24) -> 16개 레이어
    - [16, 32]: [16, 32) -> 16개 레이어
    - [16, 32]: [16, 32) -> 16개 레이어 


```python
peft_model = get_peft_model(model, config)
add_model_param = sum(p.numel() for p in peft_model.parameters()) # 7,283,675,136

len(peft_model.base_model.model.model.layers)    # (0-63) 64 layers

peft_model
```

```
PeftModel(
  (base_model): LoraModel(
    (model): MistralForCausalLM(
      (model): MistralModel(
        (embed_tokens): Embedding(32000, 4096)
        (layers): ModuleList(
          (0-63): 64 x MistralDecoderLayer(
            (self_attn): MistralSdpaAttention(
                ...
```

- layer replication 기능을 통해 약 14B의 모델을 생성

## ref

- [peft-0.10.0](https://github.com/huggingface/peft/releases/tag/v0.10.0)
- [layer-replication-with-lora](https://huggingface.co/docs/peft/developer_guides/lora#memory-efficient-layer-replication-with-lora)