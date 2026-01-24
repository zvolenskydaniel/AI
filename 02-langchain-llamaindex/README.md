# LangChain & LlamaIndex

## Overview
The previous chapter introduced the fundamentals of integrating Large Language Models (LLMs) into Python applications using direct API calls. While this approach is sufficient for simple use cases, real-world AI applications often require more structure, composability, and control.

This chapter introduces **LangChain** and **LlamaIndex**, two open-source frameworks designed to help build more complex LLM-powered systems. These frameworks provide higher-level abstractions for composing prompts, managing context and memory, integrating external tools and data sources, and orchestrating multi-step workflows.

Rather than replacing direct API usage, LangChain and LlamaIndex build on top of it, enabling scalable and maintainable AI application architectures.

## Goal
The goal of this chapter is to understand how higher-level frameworks can be used to design structured, modular, and production-oriented AI systems.

By the end of this chapter, the focus is on moving from single prompt–response interactions toward applications that combine multiple steps, external data, and reusable components.

## Core Concepts
- [Chains, tools, and memory](#chains-tools-and-memory)
- [Retrieval-Augmented Generation (RAG)]()
- [Document indexing and querying]
- [Modular AI pipelines]()
- [Why Frameworks Are Needed]()
- [Summary]() 

## Chains, tools, and memory

### Conceptual Overview
After learning how to call an LLM directly via an API (Chapter 1), the next challenge is **orchestration**:
- How do we break a task into multiple steps?
- How do we combine LLM reasoning with external systems (APIs, databases, tools)?
- How do we preserve context across multiple interactions?

This is exactly the problem space that **LangChain** was designed to solve.

**LangChain** is a framework with a modular and flexible set of tools for building a wide range of **NLP applications**. It offers a standard interface for constructing chains, extensive integrations with various tools, and complete end-to-end chains for common application scenarios.

> ***NLP applications** allow computers to understand and process human language, with common examples including virtual assistants like Siri and Alexa, chatbots for customer service, and tools for sentiment analysis on social media. Other applications include automatic translation, spam filtering, text summarization, and analyzing large volumes of text for business insights or security threats.*

At a high level:
- *chains* define how LLM calls are connected together
- *tools* let LLMs act outside of pure text generation
- *memory* allows LLM-based applications to remember past interactions

### 1. Chains
A **chain** is a sequence of operations where:
- inputs flow through one or more steps
- each step may involve an LLM call, a prompt template, or a transformation
- the output of one step feeds into the next

Think of a chain as a **pipeline for reasoning**. Example use cases:
- generate → summarize → format
- translate → analyze sentiment → extract entities
- ask a question → refine it → answer with context

#### Minimal LangChain Chain Example
```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage

# Load .env
load_dotenv()

# Initialize model
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.2
)

# Input/User data
COUNTRY = "Slovakia"
PROMPT = f"What's capital of {COUNTRY}?"

# Define System and Human messages
messages = [
    SystemMessage(content = "You are a helpful assistant."),
    HumanMessage(content = PROMPT),
]

# Invoke returns an AIMessage
ai_message = llm.invoke(messages)

# Print model output
print(ai_message.content)

```

### 2. Tools
In *LangChain*, a **tool** is a callable interface (*typically a Python function*) that acts as a specialized skill for an AI agent, allowing it to interact with the outside world, execute code, fetch real-time data, or call APIs. Tools enable LLMs to move beyond text generation by providing structured inputs and receiving outputs to perform actions.

Tools allow LLMs to:
- fetch real-time data
- query databases
- perform calculations
- trigger business logic

> *The LLM decides **when** and **how** to use the tool.*

#### Conceptual Tool Example
```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent

# Load .env
load_dotenv()

# Return country population
@tool
def get_country_population(country: str) -> str:
    data = {
        "Slovakia": "5.4 million",
        "Germany": "83 million"
    }
    return data.get(country, "Unknown")

# Initialize model
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.2
)

# Input/User data
COUNTRY = "Slovakia"
PROMPT = f"What's country population of {COUNTRY}?"

# Create the Agent
agent = create_agent(
    model = llm,
    tools = [get_country_population]
)

# Run it
ai_message = agent.invoke({
    "messages": [
        SystemMessage(content = "You are a helpful assistant."),
        HumanMessage(content = PROMPT)
    ]
})

# Print model output
print(ai_message["messages"][-1].content)

```

When registered as a tool:
- the LLM can call get_country_population("Slovakia")
- receive the result
- use it to generate a final response

This is how LLMs move from *text-only* to **action-capable systems**.

### 3. Memory
**Memory** is a system that remembers information about previous interactions. For AI agents, memory is crucial because it lets them remember previous interactions, learn from feedback, and adapt to user preferences. As agents tackle more complex tasks with numerous user interactions, this capability becomes essential for both efficiency and user satisfaction. Short term memory lets your application remember previous interactions within a single thread or conversation.

Without memory:
- every request is stateless
- the model forgets previous interactions

With memory:
- conversations feel continuous
- applications can track preferences, history, or prior answers

AI applications need memory to share context across multiple interactions.
- short-term memory as a part of your agent’s state to enable multi-turn conversations
- long-term memory to store user-specific or application-level data across sessions

#### Manual Short-Term Memory Example
- `manual_shortterm_memory.py`
```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load .env
load_dotenv()

# Initialize model
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.2
)

# Create empty list to store conversation history
history = []  # list of (role, message) tuples

# Build prompt template that allows inserting history
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content = "You are a helpful assistant who respond in max. 4 words."),
    MessagesPlaceholder(variable_name = "history"),
    HumanMessage(content = "{user_input}")
])

# Input/User data
user_inputs = ["My name is Daniel and I love biking.", "What is my favourite activity?"]

for user_text in user_inputs:
    # print user's input
    print("User:", user_text)

    # Append user message to history
    history.append(HumanMessage(content = user_text))

    # Format prompt with messages
    prompt_messages = prompt.format_messages(
        history = history,
        user_input = user_text
    )

    # Invoke LLM
    response = llm.invoke(prompt_messages)

    # Get response content
    content = response.content if hasattr(response, "content") else str(response)
    print("Assistant:", content)

    # Append assistant message to history
    history.append(AIMessage(content = response.content))

```

In a manual short-term memory approach, the entire conversation between the user and the AI assistant is stored **in memory (RAM)** during runtime.

This type of memory:
- exists only for the duration of the session
- is lost as soon as the application terminates or the conversation ends
- is simple to implement and suitable for temporary context handling

Short-term memory is commonly used for:
- single-session chat applications
- interactive scripts
- prototyping and experimentation

#### Manual Long-Term Memory Example
- `history.json`
```txt
[
    {
        "role": "human",
        "content": "My name is Daniel and I love biking."
    }
]
```

- `manual_longterm_memory.py`
```python
import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load .env
load_dotenv()

# Initialize model
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.2
)

# Get the directory where the script itself is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Combine that directory with the filename
file_path = os.path.join(script_dir, "history.json")

# Create empty list to store conversation history
history = []  # list of (role, message) tuples

# Get previous conversations: read the file history.json
with open(file_path, "r") as file:
    data = json.load(file)
    history = [
        HumanMessage(content = item["content"]) if item["role"] == "human"
        else AIMessage(content = item["content"])
        for item in data
    ]

# Build prompt template that allows inserting history
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content = "You are a helpful assistant who respond in max. 4 words."),
    MessagesPlaceholder(variable_name = "history"),
    HumanMessage(content = "{user_input}")
])

# Input/User data
user_inputs = ["What is my name?", "What is my favourite activity?", "quit"]

for user_text in user_inputs:
    # Print user's input
    print("User:", user_text)

    # Capture the conversation into history.json
    if user_text.lower() in ("exit", "quit", "q"):
        serializable = [{"role": msg.type, "content": msg.content} for msg in history]
        with open(file_path, "w") as file:
            json.dump(serializable, file, indent=4)
        print("Conversation saved. Bye!")
        break

    # Append user message to history
    history.append(HumanMessage(content = user_text))

    # Format prompt with messages
    prompt_messages = prompt.format_messages(
        history = history,
        user_input = user_text
    )

    # Invoke LLM
    response = llm.invoke(prompt_messages)

    # Get response content
    content = response.content if hasattr(response, "content") else str(response)
    print("Assistant:", content)

    # Append assistant message to history
    history.append(AIMessage(content = response.content))

```

In a manual long-term memory approach, the conversation between the user and the AI assistant is **persisted to local storage**, typically in a structured format such as a `JSON file`.

Each interaction is stored using a predefined schema that preserves:
- the original `HumanMessage`
- the original `AIMessage`

Because the structure of the messages is maintained, previously stored conversations can be:
- reloaded at application startup
- rehydrated back into message objects
- used as historical context for future interactions

This enables the assistant to retain knowledge across sessions, effectively simulating long-term memory.

#### Memory Growth and Summarization

As long-term memory grows over time, storing the full conversation history may become inefficient or impractical.

A common optimization strategy is **memory summarization**, where:
- older conversation segments are periodically summarized
- detailed message history is replaced with concise semantic summaries
- the overall context is preserved while significantly reducing storage size

This approach allows long-term memory to scale while remaining performant and cost-efficient.

> *These manual memory strategies help clarify how memory works conceptually, before introducing framework-provided memory abstractions such as those in LangChain.*

## Retrieval-Augmented Generation (RAG)
TBD

## Document indexing and querying
TBD

## Modular AI pipelines
TBD

## Why Frameworks Are Needed
As AI applications grow in complexity, challenges such as context management, prompt reuse, tool integration, and data retrieval become difficult to handle with ad-hoc code. Frameworks like **LangChain** and **LlamaIndex** address these challenges by providing standardized abstractions and patterns.

## Summary
TBD

## What’s Next
The next section Automation & Agent-Based Systems focuses on:
- single-agent vs multi-agent design
- CrewAI and AutoGen fundamentals
- task orchestration and delegation
- failure handling and observability
