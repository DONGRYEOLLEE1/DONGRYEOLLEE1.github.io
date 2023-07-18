---
layout: post
title: Daily of Developing [7/17]
subtitle: Docker, Text-generation Inference
tags: [NLP, Finetuning, Docker]
categories: Finetuning
use_math: true
comments: true
published: true
---

# Docker

- 컨테이너 확인
```docker 
$ docker ps -a
```

- 컨테이너 제거
```docker
$ docker stop <container ID> 또는
$ docker rm <container ID>  또는

$ docker container prune
```

- 용량 이슈 대처
  1. 모든 컨테이너 삭제, Active 되어있는 모든 컨테이너 중지
  2. 파일 경로 수동 지정, `/var/docker/daemon.json`

```
{
    "data-root" : "/home/user/setup/docker-data"
}
```

```docker
$ docker info
```

![Alt text](/img/docker.png)


# TGI

- `EleutherAI/polyglot-ko-12.8b` 모델 Tensor Parallel serving 성공 (PoC)
- Local LLM (LoRA) 파일 적용 가능한지 알아보다가 LoRA 모델은 아직 지원하지 않음을 알아내었고 후속조치 어떻게 해야할지 고민

![Alt text](/img/tgi1.png)

![Alt text](/img/tgi1.png)

- `merge_and_unload`를 통해 LoRa 모델을 basemodel로 merge. 이렇게 되면 LoRa weight를 잃게 되는게 아닐까?

```python
# peft/tuners/lora.py - 601 line

def merge_and_unload(self):
    r"""
    This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model as a standalone model.

    Example:

    >>> from transformers import AutoModelForCausalLM
    >>> from peft import PeftModel

    >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
    >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
    >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
    >>> merged_model = model.merge_and_unload()
    """
    return self._unload_and_optionally_merge()

# peft/tuners/lora.py - 438 line

def _unload_and_optionally_merge(self, merge=True):
    if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
        raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

    key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
    for key in key_list:
        try:
            parent, target, target_name = _get_submodules(self.model, key)
        except AttributeError:
            continue
        if isinstance(target, LoraLayer):
            if isinstance(target, nn.Embedding):
                new_module = torch.nn.Embedding(target.in_features, target.out_features)
            elif isinstance(target, nn.Conv2d):
                new_module = torch.nn.Conv2d(
                    target.in_channels,
                    target.out_channels,
                    kernel_size=target.kernel_size,
                    stride=target.stride,
                    padding=target.padding,
                    dilation=target.dilation,
                )
            else:
                bias = target.bias is not None
                if getattr(target, "is_target_conv_1d_layer", False):
                    new_module = Conv1D(target.out_features, target.in_features)
                else:
                    new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
            if merge:
                target.merge()
            self._replace_module(parent, target_name, new_module, target)

        # save any additional trainable modules part of `modules_to_save`
        if isinstance(target, ModulesToSaveWrapper):
            setattr(parent, target_name, target.modules_to_save[target.active_adapter])

    return self.model

# peft/tuners/lora.py - 361 line

def _replace_module(self, parent_module, child_name, new_module, old_module):
    setattr(parent_module, child_name, new_module)
    new_module.weight = old_module.weight
    if hasattr(old_module, "bias"):
        if old_module.bias is not None:
            new_module.bias = old_module.bias

    if getattr(old_module, "state", None) is not None:
        new_module.state = old_module.state
        new_module.to(old_module.weight.device)

    # dispatch to correct device
    for name, module in new_module.named_modules():
        if "lora_" in name:
            module.to(old_module.weight.device)
        if "ranknum" in name:
            module.to(old_module.weight.device)

```

- `_replace_module`을 통해 weight와 bias를 그대로 가져옴을 확인. Embedding layer에 weight 넣는 형태

# Ref.

- [#482](https://github.com/huggingface/text-generation-inference/issues/482)
- [#615](https://github.com/huggingface/peft/issues/615)
- [text-generation-inference](https://github.com/huggingface/text-generation-inference)
- [peft-tuners](https://huggingface.co/docs/peft/main/en/package_reference/tuners#peft.LoraModel.merge_and_unload)