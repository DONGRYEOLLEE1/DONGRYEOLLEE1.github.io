---
layout: post
title: Daily of Developing [7/29]
subtitle: LoRA, bitsandbytes
tags: [Finetuning, LoRA]
categories: Finetuning
use_math: true
comments: true
published: true
---

# Error

> undefined symbol: cget_col_row_stats


# Solution

```bash
$ cd {ENV}/lib/python-3.10/site-packages/bitsandbytes
$ cp libbitsandbytes_cuda118.so libbitsandbytes_cpu.so
```

- cuda 버전에 맞게 파일을 복사해주면 됨


# Ref

[#156](https://github.com/TimDettmers/bitsandbytes/issues/156)