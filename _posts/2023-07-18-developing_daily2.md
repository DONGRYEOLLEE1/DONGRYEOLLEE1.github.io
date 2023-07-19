---
layout: post
title: Daily of Developing [7/19]
subtitle: Gradio, TGI
tags: [NLP, Finetuning, Gradio]
categories: Finetuning
use_math: true
comments: true
published: true
---

## Gradio

- `Stream`

1. `Gradio` : `transformers.TextIteratorStreamer`

```python
def generate(user_message):
    prompt = f"### 질문: {user_message}\n\n### 답변:"
    model_inputs = tokenizer([prompt], return_tensors = 'pt', return_token_type_ids = False).to('cuda')

    streamer = TextIteratorStreamer(tokenizer, timeout = 100, skip_prompt = True, skip_special_tokens = True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        temperature= 0.7,
        top_p=0.95,
        top_k=50,
        max_new_tokens=512,
        do_sample=True,
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    ...
```

2. `TGI` : `text-generation.Client.generate_stream`

```python
def generate(...):
    ...

    stream = client.generate_stream(
            prompt,
            **generate_kwargs,
        )

    output = ""
    for idx, response in enumerate(stream):
        if response.token.text == '':
            break

        if response.token.special:
            continue
        output += response.token.text
        if idx == 0:
            history.append(" " + output)
        else:
            history[-1] = output

        chat = [(history[i].strip(), history[i + 1].strip()) for i in range(0, len(history) - 1, 2)]

        yield chat, history, user_message, ""

    return chat, history, user_message, ""
```

## TS

- `grdio`를 통해 서비스 후, public URL은 접속이 되었지만 `local URL`은 접속 x
  - `port 7860` : Port-fowarding 진행 후, 재접속하였으나 접속 x
  - `local ip`로 접속하였으나 x

- Solution
  - `server_name="0.0.0.0"`을 통해 모든 IP에서 접속 가능하게 만들어주기

```python
with gr.Blocks() as demo:
    ...

demo.queue().launch(server_name = "0.0.0.0")
```

## ref

- [gradio-docs](https://www.gradio.app/docs/interface)
- [joaogante/transformers_streaming](https://huggingface.co/spaces/joaogante/transformers_streaming/blob/main/app.py#L67)
- [uwnlp/guanaco-playground-tgi](https://huggingface.co/spaces/uwnlp/guanaco-playground-tgi/blob/main/app.py)
- [How to share gradio app in my local machine](https://discuss.huggingface.co/t/how-to-share-gradio-app-in-my-local-machine/37979)