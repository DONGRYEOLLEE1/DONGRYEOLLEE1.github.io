---
layout: post
title: (vLLM) When loading the weights, occurs infinite loading problem
subtitle: 
tags: [python, vLLM]
categories: python
use_math: true
comments: true
published: true
---


# problem

- `tensor-parallel-size` >= 2 지정시, weight 무한 로딩 문제

```bash
export MODEL_NAME="..."
export OPENAI_API_KEY="..."
vllm serve $MODEL_NAME -tp 2
```

```bash
...

(VllmWorkerProcess pid=3230) INFO 10-08 07:16:13 multiproc_worker_utils.py:215] Worker ready; awaiting tasks
(VllmWorkerProcess pid=3230) INFO 10-08 07:16:13 utils.py:784] Found nccl from library libnccl.so.2
INFO 10-08 07:16:13 utils.py:784] Found nccl from library libnccl.so.2
INFO 10-08 07:16:13 pynccl.py:63] vLLM is using nccl==2.20.5
(VllmWorkerProcess pid=3230) INFO 10-08 07:16:13 pynccl.py:63] vLLM is using nccl==2.20.5
INFO 10-08 07:16:14 custom_all_reduce_utils.py:232] reading GPU P2P access cache from /home/bigster/.cache/vllm/gpu_p2p_access_cache_for_0,1.json
(VllmWorkerProcess pid=3230) INFO 10-08 07:16:14 custom_all_reduce_utils.py:232] reading GPU P2P access cache from /home/bigster/.cache/vllm/gpu_p2p_access_cache_for_0,1.json
INFO 10-08 07:16:14 shm_broadcast.py:241] vLLM message queue communication handle: Handle(connect_ip='127.0.0.1', local_reader_ranks=[1], buffer=<vllm.distributed.device_communicators.shm_broadcast.ShmRingBuffer object at 0x71fcf35674f0>, local_subscribe_port=52931, local_sync_port=59409, remote_subscribe_port=None, remote_sync_port=None)
INFO 10-08 07:16:14 model_runner.py:680] Starting to load model /data/models/Meta-Llama-3.1-8B-Instruct/...
(VllmWorkerProcess pid=3230) INFO 10-08 07:16:14 model_runner.py:680] Starting to load model /data/models/Meta-Llama-3.1-8B-Instruct/...
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:02<00:07,  2.60s/it]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:17<00:19, 10.00s/it]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:32<00:11, 11.98s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:48<00:00, 13.53s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:48<00:00, 12.01s/it]

INFO 10-08 07:17:03 model_runner.py:692] Loading model weights took 7.5122 GB
(VllmWorkerProcess pid=3230) INFO 10-08 07:17:03 model_runner.py:692] Loading model weights took 7.5122 GB
```

# solution

- 재부팅

```bash
sudo reboot
```

# reference

- [Debugging hang/crash issues](https://docs.vllm.ai/en/stable/getting_started/debugging.html#debugging-hang-crash-issues)
- [vllm-github-issue-#5062](https://github.com/vllm-project/vllm/issues/5062)