---
layout: post
title: 
subtitle: 
tags: [LangGraph]
categories: Developing
use_math: true
comments: true
published: true
---

## Environment

- python: 3.13
- langchain: 0.3.26
- langgraph: 0.5.3

## BaseLine


```python
# util.py

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class Category(str, Enum):
    WEB = "웹검색"
    NORMAL = "일반질문"

class ResponseFormatter(BaseModel):
    """사용자의 질문 분류 결과"""
    question: str = Field(description = "사용자의 질문")
    classification: Category = Field(description = "질문의 분류 결과")
```

```python
# app.py

route_llm_with_tools = route_llm.with_structured_output(ResponseFormatter)
route_prompt = ChatPromptTemplate.from_messages([
    ("system", route_template),
    ("human", "다음 질문을 목적에 맞게 분류하세요. {question}")])
route_chain = route_prompt | route_llm_with_tools

chain = (
    {
        "context": web_ret | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt 
    | llm
)

normal_chain = normal_prompt | llm

class CustomState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    

def routing(state: CustomState) -> str:
    
    route_response = route_chain.invoke({"question": state['messages'][-1].content})
    route_cls = route_response.classification.value
    
    return route_cls
    
def api_generate(state: CustomState):
    # print("API GENERATE NODE!!")
    
    response = normal_chain.invoke(state['messages'])
    
    return {"messages": response}
    

def web_generate(state: CustomState):
    # print("WEB GENERATE NODE!!")
    # print(f"StateMessages: {state['messages']}")
    question = state['messages'][-1].content
    response = chain.invoke(question)
    return {"messages": response}
    
workflow = StateGraph(CustomState)
workflow.add_node("web_generate", web_generate)
workflow.add_node("api_generate", api_generate)
workflow.add_conditional_edges(
    START,
    routing,
    {
        "웹검색": "web_generate",
        "일반질문": "api_generate"
    }
)
workflow.add_edge("web_generate", END)
workflow.add_edge("api_generate", END)
graph = workflow.compile(checkpointer = memory)

# streaming 함수
def stream_inference(query: str):
    inputs = [HumanMessage(content = query)]
    for msg, metadata in graph.stream(
        {"messages": inputs}, stream_mode = "messages", config = {"configurable": {"thread_id": "11111"}}):
        if metadata.get("langgraph_node") in ("web_generate", "api_generate"):
            if msg.content and not isinstance(msg, HumanMessage):
                yield msg.content


# streaming iterator
for chunk in stream_inference("hi"):
  print(chunk)
```

## Problem

- `Routing` 노드에서의 결과값(`ResponseFormatter.classification.value`)을 추론레벨에서 별도의 변수로 선언 필요
- 더불어서 `graph.stream(...)` 레벨에서의 리턴 값을 면밀하게 분석

## 구조 분석

```python
# msg 값 출력
content='' additional_kwargs={} response_metadata={} id='run--29893c55-d6bc-4e16-ab8a-4512a05ea313'
content='{\n\n\n' additional_kwargs={} response_metadata={} id='run--29893c55-d6bc-4e16-ab8a-4512a05ea313'
content=' ' additional_kwargs={} response_metadata={} id='run--29893c55-d6bc-4e16-ab8a-4512a05ea313'
content=' "' additional_kwargs={} response_metadata={} id='run--29893c55-d6bc-4e16-ab8a-4512a05ea313'
...
content='' additional_kwargs={'parsed': ResponseFormatter(question='hi', classification=<Category.NORMAL: '일반질문'>), 'refusal': None} response_metadata={'token_usage': None, 'model_name': '/data/models/Qwen3-14B-AWQ/', 'system_fingerprint': None, 'id': 'chatcmpl-3b9cd97d04ec40bb8ffe2ae0d2d68546', 'service_tier': None} id='run--29893c55-d6bc-4e16-ab8a-4512a05ea313'
...
```

```python
def stream_inference(query: str):
  for msg, metadata in graph.stream(...):
    ...

    # routing 노드의 결과값 전달
    if msg.additional_kwargs.get("parsed") != None:
      intent = msg.additional_kwargs.get("parsed").classification.value
      print(f"질문 의도: {intent}")
```

```python
# 구조분석을 위한 metadata 출력
{'thread_id': '11111', 'langgraph_step': 42, 'langgraph_node': '__start__', 'langgraph_triggers': ('__start__',), 'langgraph_path': ('__pregel_pull', '__start__'), 'langgraph_checkpoint_ns': '__start__:cae7c5dd-b89b-04dc-25d0-bb7aa4149567', 'checkpoint_ns': '__start__:cae7c5dd-b89b-04dc-25d0-bb7aa4149567', 'ls_provider': 'openai', 'ls_model_name': '/data/models/...', 'ls_model_type': 'chat', 'ls_temperature': None, 'tags': ['langsmith:hidden']}
{'thread_id': '11111', 'langgraph_step': 42, 'langgraph_node': '__start__', 'langgraph_triggers': ('__start__',), 'langgraph_path': ('__pregel_pull', '__start__'), 'langgraph_checkpoint_ns': '__start__:cae7c5dd-b89b-04dc-25d0-bb7aa4149567', 'checkpoint_ns': '__start__:cae7c5dd-b89b-04dc-25d0-bb7aa4149567', 'ls_provider': 'openai', 'ls_model_name': '/data/models/...', 'ls_model_type': 'chat', 'ls_temperature': None, 'tags': ['langsmith:hidden']}
...
{'thread_id': '11111', 'langgraph_step': 43, 'langgraph_node': 'api_generate', 'langgraph_triggers': ('branch:to:api_generate',), 'langgraph_path': ('__pregel_pull', 'api_generate'), 'langgraph_checkpoint_ns': 'api_generate:9f4b362d-ce76-0452-4202-65c5e6ddb264', 'checkpoint_ns': 'api_generate:9f4b362d-ce76-0452-4202-65c5e6ddb264', 'ls_provider': 'openai', 'ls_model_name': '/data/models/...', 'ls_model_type': 'chat', 'ls_temperature': None}
안  # msg.content
{'thread_id': '11111', 'langgraph_step': 43, 'langgraph_node': 'api_generate', 'langgraph_triggers': ('branch:to:api_generate',), 'langgraph_path': ('__pregel_pull', 'api_generate'), 'langgraph_checkpoint_ns': 'api_generate:9f4b362d-ce76-0452-4202-65c5e6ddb264', 'checkpoint_ns': 'api_generate:9f4b362d-ce76-0452-4202-65c5e6ddb264', 'ls_provider': 'openai', 'ls_model_name': '/data/models/...', 'ls_model_type': 'chat', 'ls_temperature': None}
녕  # msg.content
```