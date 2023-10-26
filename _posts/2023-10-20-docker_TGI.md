---
layout: post
title: Implementation on Text-Generation Inference (TGI)
subtitle: 
tags: [Docker, TGI]
categories: Developing
use_math: true
comments: true
published: true
---

# Env

- OS :  Ubuntu 20.04
- Docker : 24.0.6


# 1

- TGI에 모델을 적재시키기 위해선 `lora` 파인튜닝 모델 -> standalone 모델 변환(Full-finetuning 모델 필요X)

```python

def main(
    base_model_id: str,
    lora_model_id: str,
    output_dir: str
    ):
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map = 'auto')
    
    try:
        config = PeftConfig.from_pretrained(lora_model_id)
        lora_model = get_peft_model(base_model, config)
        
    except RuntimeError as re:
        lora_model = PeftModel.from_pretrained(
            base_model, 
            lora_model_id, 
            device_map = {"" : "cpu"})
    
    base_vocab_size = base_model.get_input_embeddings().weight.size(0)
    
    print(f"Base Model vocab size : {base_vocab_size}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    if base_vocab_size != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"Resizing Vocabulary to {len(tokenizer)}")
        
    base_first_weight = base_model.gpt_neox.layers[0].attention.query_key_value.weight.clone()
    lora_first_weight = lora_model.gpt_neox.layers[0].attention.query_key_value.weight.clone()
    
    assert torch.allclose(base_first_weight, lora_first_weight)
    
    
    standalone_model = lora_model.merge_and_unload()
    standalone_model.save_pretrained(output_dir, safe_serialization = True)
    tokenizer.save_pretrained(output_dir)
    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument('--base_model_id',
                        type = 'str',
                        default = "base_model_id")
    parser.add_argument('--lora_model_id',
                        type = str,
                        default = "lora_model_id")
    parser.add_argument('--output_dir',
                        type = str,
                        default = "/model_file/OUTPUT_DIRECTORY/")
    
    args = parser.parse_args()
    
    main(
        base_model_id = args.base_model_id,
        peft_model_id = args.lora_model_id,
        output_dir = args.output_dir
    )
```

```docker
$ cd model_file
$ nohup sudo docker run --gpus all --shm-size 1g -p 8080:80 --volume /DATA:/data ghcr.io/huggingface/text-generation-inference:latest --model-id /data/OUTPUT_DIRECTORY --num-shard {NUMBER_OF_GPUS} --rope-scaling dynamic --rope-factor 2.0 &

2023-10-26T06:37:55.196641Z  INFO shard-manager: text_generation_launcher: Shard ready in 12.91406831s rank=1
2023-10-26T06:37:55.292255Z  INFO text_generation_launcher: Starting Webserver
2023-10-26T06:37:55.351346Z  WARN text_generation_router: router/src/main.rs:194: no pipeline tag found for model /data/OUTPUT_DIRECTORY
2023-10-26T06:37:55.365262Z  INFO text_generation_router: router/src/main.rs:213: Warming up model
2023-10-26T06:37:56.919473Z  INFO text_generation_router: router/src/main.rs:246: Setting max batch total tokens to 82224
2023-10-26T06:37:56.919503Z  INFO text_generation_router: router/src/main.rs:247: Connected
2023-10-26T06:37:56.919516Z  WARN text_generation_router: router/src/main.rs:252: Invalid hostname, defaulting to 0.0.0.0
```

# Reference

- [TGI-github](https://github.com/huggingface/text-generation-inference/issues)
- [TGI-huggingface](https://huggingface.co/docs/text-generation-inference/index)