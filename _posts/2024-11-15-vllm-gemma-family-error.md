---
layout: post
title: vLLM error - gemma계열 모델에서 발생하는 오류
subtitle: flashinfer
tags: [NLP, vllm]
categories: Developing
use_math: true
comments: true
published: true
---

## Env

- os: Ubuntu 22.04
- nvidia driver version: 535.183.01
- cuda: 11.8
- vllm version: 0.5.3
- torch version: 2.3.1
- transformers version: 4.45.2

## Error

- vllm을 통해 `gemma-2-27b-it` 모델 load시 아래와 같은 에러 발생

```
...

(VllmWorkerProcess pid=239160) INFO 11-15 04:16:07 model_runner.py:680] Starting to load model /data/models/gemma-2-27b-it/...
Loading safetensors checkpoint shards:   0% Completed | 0/12 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:   8% Completed | 1/12 [00:15<02:45, 15.06s/it]
Loading safetensors checkpoint shards:  17% Completed | 2/12 [00:30<02:33, 15.30s/it]
Loading safetensors checkpoint shards:  25% Completed | 3/12 [00:45<02:18, 15.34s/it]
Loading safetensors checkpoint shards:  33% Completed | 4/12 [01:02<02:05, 15.70s/it]
Loading safetensors checkpoint shards:  42% Completed | 5/12 [01:17<01:48, 15.54s/it]
Loading safetensors checkpoint shards:  50% Completed | 6/12 [01:36<01:40, 16.69s/it]
Loading safetensors checkpoint shards:  58% Completed | 7/12 [01:47<01:13, 14.74s/it]
Loading safetensors checkpoint shards:  67% Completed | 8/12 [02:05<01:04, 16.02s/it]
Loading safetensors checkpoint shards:  75% Completed | 9/12 [02:08<00:35, 11.74s/it]
Loading safetensors checkpoint shards:  83% Completed | 10/12 [02:13<00:19,  9.84s/it]
Loading safetensors checkpoint shards:  92% Completed | 11/12 [02:22<00:09,  9.51s/it]
Loading safetensors checkpoint shards: 100% Completed | 12/12 [02:23<00:00,  6.82s/it]
Loading safetensors checkpoint shards: 100% Completed | 12/12 [02:23<00:00, 11.93s/it]

(VllmWorkerProcess pid=239160) INFO 11-15 04:18:31 model_runner.py:692] Loading model weights took 25.4489 GB
INFO 11-15 04:18:31 model_runner.py:692] Loading model weights took 25.4489 GB
[rank0]: Traceback (most recent call last):
[rank0]:   File "/data/envs/llm_test/bin/vllm", line 8, in <module>
[rank0]:     sys.exit(main())
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/scripts.py", line 148, in main
[rank0]:     args.dispatch_function(args)
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/scripts.py", line 28, in serve
[rank0]:     run_server(args)
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/entrypoints/openai/api_server.py", line 231, in run_server
[rank0]:     if llm_engine is not None else AsyncLLMEngine.from_engine_args(
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py", line 466, in from_engine_args
[rank0]:     engine = cls(
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py", line 380, in __init__
[rank0]:     self.engine = self._init_engine(*args, **kwargs)
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py", line 547, in _init_engine
[rank0]:     return engine_class(*args, **kwargs)
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/engine/llm_engine.py", line 265, in __init__
[rank0]:     self._initialize_kv_caches()
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/engine/llm_engine.py", line 364, in _initialize_kv_caches
[rank0]:     self.model_executor.determine_num_available_blocks())
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/executor/distributed_gpu_executor.py", line 38, in determine_num_available_blocks
[rank0]:     num_blocks = self._run_workers("determine_num_available_blocks", )
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/executor/multiproc_gpu_executor.py", line 178, in _run_workers
[rank0]:     driver_worker_output = driver_worker_method(*args, **kwargs)
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
[rank0]:     return func(*args, **kwargs)
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/worker/worker.py", line 179, in determine_num_available_blocks
[rank0]:     self.model_runner.profile_run()
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
[rank0]:     return func(*args, **kwargs)
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/worker/model_runner.py", line 888, in profile_run
[rank0]:     model_input = self.prepare_model_input(
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/worker/model_runner.py", line 1225, in prepare_model_input
[rank0]:     model_input = self._prepare_model_input_tensors(
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/worker/model_runner.py", line 801, in _prepare_model_input_tensors
[rank0]:     return builder.build()  # type: ignore
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/worker/model_runner.py", line 523, in build
[rank0]:     attn_metadata = self.attn_metadata_builder.build(
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/attention/backends/flash_attn.py", line 286, in build
[rank0]:     raise ValueError(
[rank0]: ValueError: Please use Flashinfer backend for models with logits_soft_cap (i.e., Gemma-2). Otherwise, the output might be wrong. Set Flashinfer backend by export VLLM_ATTENTION_BACKEND=FLASHINFER.
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226] Exception in worker VllmWorkerProcess while processing method determine_num_available_blocks: Please use Flashinfer backend for models with logits_soft_cap (i.e., Gemma-2). Otherwise, the output might be wrong. Set Flashinfer backend by export VLLM_ATTENTION_BACKEND=FLASHINFER., Traceback (most recent call last):
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/executor/multiproc_worker_utils.py", line 223, in _run_worker_process
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]     output = executor(*args, **kwargs)
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]     return func(*args, **kwargs)
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/worker/worker.py", line 179, in determine_num_available_blocks
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]     self.model_runner.profile_run()
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]     return func(*args, **kwargs)
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/worker/model_runner.py", line 888, in profile_run
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]     model_input = self.prepare_model_input(
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/worker/model_runner.py", line 1225, in prepare_model_input
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]     model_input = self._prepare_model_input_tensors(
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/worker/model_runner.py", line 801, in _prepare_model_input_tensors
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]     return builder.build()  # type: ignore
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/worker/model_runner.py", line 523, in build
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]     attn_metadata = self.attn_metadata_builder.build(
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/attention/backends/flash_attn.py", line 286, in build
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226]     raise ValueError(
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226] ValueError: Please use Flashinfer backend for models with logits_soft_cap (i.e., Gemma-2). Otherwise, the output might be wrong. Set Flashinfer backend by export VLLM_ATTENTION_BACKEND=FLASHINFER.
(VllmWorkerProcess pid=239160) ERROR 11-15 04:18:31 multiproc_worker_utils.py:226] 
INFO 11-15 04:18:31 multiproc_worker_utils.py:123] Killing local vLLM worker processes
Fatal Python error: _enter_buffered_busy: could not acquire lock for <_io.BufferedWriter name='<stdout>'> at interpreter shutdown, possibly due to daemon threads
Python runtime state: finalizing (tstate=0x000061a57ac26600)
```


## 조치 1

- Gemma family 모델은 `flashinfer` 패키지 설치 필요 -> `flashinfer` 패키지 설치
- [FlashInfer-Install-Instruction](https://docs.flashinfer.ai/installation.html)에 따라 각 환경에 맞는 버젼 설치 필요요

```
pip install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.3/
```

## Error 2

```
(VllmWorkerProcess pid=243206) INFO 11-15 04:51:47 model_runner.py:692] Loading model weights took 25.4489 GB
[rank0]: Traceback (most recent call last):
[rank0]:   File "/data/envs/llm_test/bin/vllm", line 8, in <module>
[rank0]:     sys.exit(main())
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/scripts.py", line 148, in main
[rank0]:     args.dispatch_function(args)
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/scripts.py", line 28, in serve
[rank0]:     run_server(args)
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/entrypoints/openai/api_server.py", line 231, in run_server
[rank0]:     if llm_engine is not None else AsyncLLMEngine.from_engine_args(
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py", line 466, in from_engine_args
[rank0]:     engine = cls(
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py", line 380, in __init__
[rank0]:     self.engine = self._init_engine(*args, **kwargs)
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py", line 547, in _init_engine
[rank0]:     return engine_class(*args, **kwargs)
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/engine/llm_engine.py", line 265, in __init__
[rank0]:     self._initialize_kv_caches()
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/engine/llm_engine.py", line 364, in _initialize_kv_caches
[rank0]:     self.model_executor.determine_num_available_blocks())
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/executor/distributed_gpu_executor.py", line 38, in determine_num_available_blocks
[rank0]:     num_blocks = self._run_workers("determine_num_available_blocks", )
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/executor/multiproc_gpu_executor.py", line 178, in _run_workers
[rank0]:     driver_worker_output = driver_worker_method(*args, **kwargs)
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
[rank0]:     return func(*args, **kwargs)
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/worker/worker.py", line 179, in determine_num_available_blocks
[rank0]:     self.model_runner.profile_run()
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
[rank0]:     return func(*args, **kwargs)
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/worker/model_runner.py", line 896, in profile_run
[rank0]:     self.execute_model(model_input, kv_caches, intermediate_tensors)
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
[rank0]:     return func(*args, **kwargs)
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/worker/model_runner.py", line 1292, in execute_model
[rank0]:     model_input.attn_metadata.begin_forward()
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/attention/backends/flashinfer.py", line 146, in begin_forward
[rank0]:     self.prefill_wrapper.begin_forward(
[rank0]:   File "/data/envs/llm_test/lib/python3.10/site-packages/flashinfer/prefill.py", line 832, in plan
[rank0]:     self._wrapper.plan(
[rank0]: RuntimeError: CHECK_EQ(paged_kv_indptr.size(0), batch_size + 1) failed. 1 vs 257
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226] Exception in worker VllmWorkerProcess while processing method determine_num_available_blocks: CHECK_EQ(paged_kv_indptr.size(0), batch_size + 1) failed. 1 vs 257, Traceback (most recent call last):
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/executor/multiproc_worker_utils.py", line 223, in _run_worker_process
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]     output = executor(*args, **kwargs)
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]     return func(*args, **kwargs)
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/worker/worker.py", line 179, in determine_num_available_blocks
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]     self.model_runner.profile_run()
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]     return func(*args, **kwargs)
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/worker/model_runner.py", line 896, in profile_run
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]     self.execute_model(model_input, kv_caches, intermediate_tensors)
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]     return func(*args, **kwargs)
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/worker/model_runner.py", line 1292, in execute_model
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]     model_input.attn_metadata.begin_forward()
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/vllm/attention/backends/flashinfer.py", line 146, in begin_forward
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]     self.prefill_wrapper.begin_forward(
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]   File "/data/envs/llm_test/lib/python3.10/site-packages/flashinfer/prefill.py", line 832, in plan
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226]     self._wrapper.plan(
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226] RuntimeError: CHECK_EQ(paged_kv_indptr.size(0), batch_size + 1) failed. 1 vs 257
(VllmWorkerProcess pid=243206) ERROR 11-15 04:51:47 multiproc_worker_utils.py:226] 
[rank0]:[W CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
/usr/lib/python3.10/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 1 leaked shared_memory objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
```

## 조치 2

- `vllm` 버전 업데이트 + `flashinfer` 환경에 맞는 버전 설치 + cuda 12.4로 변경

```
pip install -U -q vllm
```

- 또는 `VLLM_ATTENTION_BACKEND=FLASHINFER` 옵션 추가해주기

## Reference

- [vllm-issue-#7060](https://github.com/vllm-project/vllm/issues/7060)
- [Detailed Steps for Running Fine-tuned Gemma-2-2b-it with vLLM](https://chenhuiyu.github.io/2024/08/07/NLP%20Insights/Running%20Fine-tuned%20Gemma-2-2b-it%20with%20vLLM/index.html)
- [vllm-requires-flashinfer--release-note-0.5.1](https://github.com/vllm-project/vllm/releases/tag/v0.5.1)