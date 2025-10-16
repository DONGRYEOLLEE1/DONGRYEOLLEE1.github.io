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
❗️❗️ Error occurred during inference. Maximum reties reached for rate limit or other error. Continuing to next test case.
❗️❗️ Test case ID: simple_3, Error: Error code: 400 - {'error': {'message': 'Encrypted content is not supported with this model.', 'type': 'invalid_request_error', 'param': 'include', 'code': None}}
Traceback (most recent call last):
  File "/data/dev/uv-test/agent-bm2/.venv/lib/python3.10/site-packages/bfcl_eval/_llm_response_generation.py", line 182, in multi_threaded_inference
    result, metadata = handler.inference(
  File "/data/dev/uv-test/agent-bm2/.venv/lib/python3.10/site-packages/bfcl_eval/model_handler/base_handler.py", line 47, in inference
    return self.inference_single_turn_FC(test_entry, include_input_log)
  File "/data/dev/uv-test/agent-bm2/.venv/lib/python3.10/site-packages/bfcl_eval/model_handler/base_handler.py", line 593, in inference_single_turn_FC
    api_response, query_latency = self._query_FC(inference_data)
  File "/data/dev/uv-test/agent-bm2/.venv/lib/python3.10/site-packages/bfcl_eval/model_handler/api_inference/openai_response.py", line 90, in _query_FC
    return self.generate_with_backoff(**kwargs)
  File "/data/dev/uv-test/agent-bm2/.venv/lib/python3.10/site-packages/tenacity/__init__.py", line 336, in wrapped_f
    return copy(f, *args, **kw)
  File "/data/dev/uv-test/agent-bm2/.venv/lib/python3.10/site-packages/tenacity/__init__.py", line 475, in __call__
    do = self.iter(retry_state=retry_state)
  File "/data/dev/uv-test/agent-bm2/.venv/lib/python3.10/site-packages/tenacity/__init__.py", line 376, in iter
    result = action(retry_state)
  File "/data/dev/uv-test/agent-bm2/.venv/lib/python3.10/site-packages/tenacity/__init__.py", line 398, in <lambda>
    self._add_action_func(lambda rs: rs.outcome.result())
  File "/usr/lib/python3.10/concurrent/futures/_base.py", line 451, in result
    return self.__get_result()
  File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
    raise self._exception
openai.BadRequestError: Error code: 400 - {'error': {'message': 'Encrypted content is not supported with this model.', 'type': 'invalid_request_error', 'param': 'include', 'code': None}}
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