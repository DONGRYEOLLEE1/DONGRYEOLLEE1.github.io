---
layout: post
title: 
subtitle: 
tags: [gorilla, BFCL, agent]
categories: Developing
use_math: true
comments: true
published: true
---

## Environment

- python: 3.10
- bfcl-eval: 2025.8.6.2

## Error

Agent performance 측정을 위한 `BFCL` 벤치마크 실행 도중, api base generation 실행시 아래의 오류 발생

```text
openai.BadRequestError: Error code: 400 - {'error': {'message': "Invalid type for 'include': expected an array of one of 'fil...lts', 'web...lts', 'mes...url', 'com...url', 'rea...ent', or 'mes...obs', but got a string instead.", 'type': 'invalid_request_error', 'param': 'include', 'code': 'invalid_type'}}
```

## Reproduction

```bash
uv run generate --model "gpt-4o-2024-11-20-FC" --test-category simple
```

## Solution

- `OpenAI.responses.create(**kwargs)`에 사용되는 `kwargs` 데이터 삽입 수정 필요
- reasoning 모델에만 `include`에 `reasoning.encrypted_content` 필요

```python
# gorilla/berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/openai_response.py 

class OpenAIResponsesHandler(BaseHandler):
    ...

    def _query_FC(...):
        ...

        kwargs = {
            "input": message,
            "model": self.model_name.replace("-FC", ""),
            "store": False,
        }

        # Only add reasoning parameters for OpenAI reasoning models
        if "o3" in self.model_name or "o4-mini" in self.model_name:
            kwargs["include"] = ["reasoning.encrypted_content"]
            kwargs["reasoning"] = {"summary": "auto"}
        else:
            kwargs["temperature"] = self.temperature

        ...

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"])}

        kwargs = {
            "input": inference_data["message"],
            "model": self.model_name.replace("-FC", ""),
            "store": False,
        }

        # Only add reasoning parameters for OpenAI reasoning models
        if "o3" in self.model_name or "o4-mini" in self.model_name:
            kwargs["include"] = ["reasoning.encrypted_content"]
            kwargs["reasoning"] = {"summary": "auto"}
        else:
            kwargs["temperature"] = self.temperature

        return self.generate_with_backoff(**kwargs)

```


## Refernce
- [opneai/guide](https://platform.openai.com/docs/guides/reasoning/how-reasoning-works?api-mode=responses#encrypted-reasoning-items)
- [OpenAI-Developer-Community-1297811](https://community.openai.com/t/using-reasoning-encrypted-content-with-background-mode/1297811/2)