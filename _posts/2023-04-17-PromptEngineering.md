---
layout: post
title: Prompt Engineering
subtitle: 
tags: [Prompt Engineering, NLP]
categories: NLP
use_math: true
comments: true
---

## Elements of Prompt

- `Instruct` : 모델에 대한 특정 task 또는 지시
- `Context` : 외부 정보나 추가적인 문장을 포함하며 이는 모델이 더 나은 응답을 하게 만들어줌
- `Input Data` : input이나 질의에 대한 사용자의 응답이나 질의를 찾아줌
- `Ouput Indicator` : 형태나 포맷을 지정


## Prompt를 지정할때 일반적인 팁

### The Instruction

`Wirte`, `Classify`, `Summarize`, `Translate`, `Order`등과 같은 command를 사용해 더 효과적이고 효율적으로 prompt를 구성할 수 있음

다양하고 환경에서 많은 경우의수를 놓고 실험하는게 최고의 방법임. 다른 키워드, 문장, 데이터와 함께 다른 instruction을 사용해 task에 맞는 최고의 성능을 뽑아내야함. 

또 prompt의 시작을 특별한 instruction으로 대체하는 방법이 있음. `###`을 사용하여 instruction과 context를 분리하는 방법임.

```
### Instruction ###
Translate the text below to Spanish:
Text: "hello!"
```

```
Hola!
```

### Avoid Impreciseness

부정확함을 회피하는방법인데 좀더 직접적이고 효과적으로 메세지를 던지는 형태임.

예를들어 prompt engineering의 컨셉을 학습하는 것에 대해 관심있다고 가정할때 아래의 것을 prompt로 넣을 수 있을 것

```
Explain the concept prompt engineering. Keep the explanation short, only a few sentences, and don't be too descriptive.
```

많은 문장이 쓰였고 비교적 확실하지 않은 문장이 쓰였음. 위의 프롬프트로도 어느 정도 좋은 반응을 얻을 수 있지만, 더 좋은 프롬프트는 매우 구체적이고 간결하며 핵심을 찌르는 것.

```
Use 2-3 sentences to explain the concept of prompt engineering to a high school student.
```

### To do or not to do?

prompt를 지정할때의 또 다른 일반적인 팁은 바로 해야되는것대신에 하지 말아야 하는 것을 말해주는 것이다. 이 방법을 통해 좀 더 구체적이고 detail에 집중된 결과를 얻을 수 있을 것이다. 

영화 추천 chatbot에 대한 예제가 있는데 정확하게 원하지 않는 것에 대해 명시하였음.

```
Prompt:
The following is an agent that recommends movies to a customer. DO NOT ASK FOR INTERESTS. DO NOT ASK FOR PERSONAL INFORMATION.
Customer: Please recommend a movie based on my interests.
Agent: 
```

```
Output:
Sure, I can recommend a movie based on your interests. What kind of movie would you like to watch? Do you prefer action, comedy, romance, or something else?
```

더 좋은 prompt를 만들어보자.

```
Prompt:
The following is an agent that recommends movies to a customer. The agent is responsible to recommend a movie from the top global trending movies. It should refrain from asking users for their preferences and avoid asking for personal information. If the agent doesn't have a movie to recommend, it should respond "Sorry, couldn't find a movie to recommend today.".
Customer: Please recommend a movie based on my interests.
Agent:
```

```
Output:
Sorry, I don't have any information about your interests. However, here's a list of the top global trending movies right now: [list of movies]. I hope you find something you like!
```


## Examples of Prompts

### QA

모델이 특정한 대답을 뱉게 하는 가장 좋은 방법중 하나는 prompt의 format을 향상시키는 것입니다. 앞서 설명한 것처럼 프롬프트는 instruction, context, 입력 및 출력 indicator를 결합하여 더 나은 결과를 얻을 수 있습니다. 이러한 구성들은 요구되어지지 않는 반면에 사용자가 원하는 더 특정한 instruction과 함께 좋은 practice가 됩니다. 아래의 예시는 좀 더 구조적은 prompt를 구성하는 방법입니다.

```
Prompt:
Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer.
Context: Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, the molecule was able to bind to the surface of T cells and limit their cell-killing potential. In 1986, it was approved to help prevent organ rejection after kidney transplants, making it the first therapeutic antibody allowed for human use.
Question: What was OKT3 originally sourced from?
Answer:
```

```
Output:
Mice.
```

### Conversation

아마 prompt engineering에 대해 사용자가 가장 원하는 것은 LLM system에 모델이 어떻게 동작할지 지시하는 것일겁니다. 이건 사용자가 customer 서비스 챗봇처럼 대화 시스템을 구성해나가는데 매우 유용할 것입니다.

예를들면, 더 기술적이고 과학적인 응답을 위한 대화형 시스템을 하나 만들어 봅시다. instruction을 통해 대답(행동)하는 방법을 특정하게 말해줄 수 있습니다. 이를 때때로 우린 `role prompting`이라고도 합니다.

```
Prompt:
The following is a conversation with an AI research assistant. The assistant tone is technical and scientific.
Human: Hello, who are you?
AI: Greeting! I am an AI research assistant. How can I help you today?
Human: Can you tell me about the creation of blackholes?
AI:
```

```
Output:
Sure! Black holes are regions of spacetime where the gravitational force is so strong that nothing, not even light, can escape from it. They are created when a very massive star dies and its core collapses in on itself, forming a singularity of infinite density. The intense gravity of the singularity pulls in all the matter and radiation around it, creating the black hole.
```

좀 더 기술적이고 전문적인 단어들로 대답이 뽑혔습니다. 좀 더 accessible한 대답으로 바꿔 봅시다.

```
Prompt:
The following is a conversation with an AI research assistant. The assistant answers should be easy to understand even by primary school students.
Human: Hello, who are you?
AI: Greeting! I am an AI research assistant. How can I help you today?
Human: Can you tell me about the creation of black holes?
AI: 
```

```
Output:
Sure! Black holes are created when a very massive star runs out of fuel and collapses in on itself. The gravity of the star is so strong that nothing, not even light, can escape its pull. This creates a point in space-time with an intense gravitational force called a black hole.
```


## Chain-of-Thought(CoT) Prompting

![CoT](https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fcot.1933d9fe.png&w=1080&q=75)

### CoT Prompting

[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)에서 소개된 CoT prompting은 복잡한 추론 기능을 추론 과정을 통해 가능케한다. Few-shot prompting과 결합하여 더 복잡한 task에서 좋은 결과를 낼 수 있다. 

```
Prompt:
The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.
A: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False.
The odd numbers in this group add up to an even number: 17,  10, 19, 4, 8, 12, 24.
A: Adding all the odd numbers (17, 19) gives 36. The answer is True.
The odd numbers in this group add up to an even number: 16,  11, 14, 4, 8, 13, 24.
A: Adding all the odd numbers (11, 13) gives 24. The answer is True.
The odd numbers in this group add up to an even number: 17,  9, 10, 12, 13, 4, 2.
A: Adding all the odd numbers (17, 9, 13) gives 39. The answer is False.
The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1. 
A:
```

```
Output:
Adding all the odd numbers (15, 5, 13, 7, 1) gives 41. The answer is False.
```

더 적은 예시를 들어줘도 맞출 수 있음!

```
Prompt:
The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.
A: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False.
The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1. 
A:
```

```
Output:
Adding all the odd numbers (15, 5, 13, 7, 1) gives 41. The answer is False.
```

### Zero-Shot CoT Prompting

![ZSCoT](https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fzero-cot.79793bee.png&w=1080&q=75)

마법의 단어 **Let's think step by step**. 각설하고 예시를 보자.

```
Prompt:
I went to the market and bought 10 apples. I gave 2 apples to the neighbor and 2 to the repairman. I then went and bought 5 more apples and ate 1. How many apples did I remain with?
```

```
Output:
11 apples
```

땡!

```
Prompt:
I went to the market and bought 10 apples. I gave 2 apples to the neighbor and 2 to the repairman. I then went and bought 5 more apples and ate 1. How many apples did I remain with?
Let's think step by step.
```

```
Output:
First, you started with 10 apples.
You gave away 2 apples to the neighbor and 2 to the repairman, so you had 6 apples left.
Then you bought 5 more apples, so now you had 11 apples.
Finally, you ate 1 apple, so you would remain with 10 apples.
```

많은 예시 필요없이 이 prompt를 통해 효과적인 방법을 낼 수 있음. 

## Reference

- [https://www.promptingguide.ai/](https://www.promptingguide.ai/)