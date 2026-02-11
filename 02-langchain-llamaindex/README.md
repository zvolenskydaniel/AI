# LangChain & LlamaIndex

## Overview
The previous chapter introduced the fundamentals of integrating Large Language Models (LLMs) into Python applications using direct API calls. While this approach is sufficient for simple use cases, real-world AI applications often require more structure, composability, and control.

This chapter introduces **LangChain** and **LlamaIndex**, two open-source frameworks designed to help build more complex LLM-powered systems. These frameworks provide higher-level abstractions for composing prompts, managing context and memory, integrating external tools and data sources, and orchestrating multi-step workflows.

Rather than replacing direct API usage, *LangChain* and *LlamaIndex* build on top of it, enabling scalable and maintainable AI application architectures.

## Goal
The goal of this chapter is to understand how higher-level frameworks can be used to design structured, modular, and production-oriented AI systems.

By the end of this chapter, the focus is on moving from single prompt–response interactions toward applications that combine multiple steps, external data, and reusable components.

## Core Concepts
- [Why Frameworks Are Needed](#why-frameworks-are-needed)
- [Chains, Tools, and Memory](#chains-tools-and-memory)
- [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
- [Document Indexing and Querying](#document-indexing-and-querying)
- [Modular AI pipelines](#modular-ai-pipelines)
- [Summary](#summary)
- [Example: AI Assistant Using Python Library LlamaIndex](https://github.com/zvolenskydaniel/AI/blob/main/02-langchain-llamaindex/ai_assistant_example.md)

## Why Frameworks Are Needed
As AI applications grow in complexity, challenges such as context management, prompt reuse, tool integration, and data retrieval become difficult to handle with ad-hoc code. Frameworks like **LangChain** and **LlamaIndex** address these challenges by providing standardized abstractions and patterns.

## Chains, Tools, and Memory

### Conceptual Overview
After learning how to call an LLM directly via an API (*Chapter 1*), the next challenge is **orchestration**:
- How do we break a task into multiple steps?
- How do we combine LLM reasoning with external systems (*APIs, databases, tools*)?
- How do we preserve context across multiple interactions?

This is exactly the problem space that **LangChain** was designed to solve.

**LangChain** is a framework with a modular and flexible set of tools for building a wide range of **NLP applications**. It offers a standard interface for constructing chains, extensive integrations with various tools, and complete end-to-end chains for common application scenarios.

> ***NLP applications** allow computers to understand and process human language, with common examples including virtual assistants like Siri and Alexa, chatbots for customer service, and tools for sentiment analysis on social media. Other applications include automatic translation, spam filtering, text summarization, and analyzing large volumes of text for business insights or security threats.*

At a high level:
- *chains* define how LLM calls are connected together
- *tools* let LLMs act outside of pure text generation
- *memory* allows LLM-based applications to remember past interactions

### Chains
A **chain** is a sequence of operations where:
- inputs flow through one or more steps
- each step may involve an LLM call, a prompt template, or a transformation
- the output of one step feeds into the next

Think of a chain as a **pipeline for reasoning**.

Example use cases:
- generate → summarize → format
- translate → analyze sentiment → extract entities
- ask a question → refine it → answer with context

#### Minimal LangChain Chain Example
```python
# minimal_langchain_chain_example.py
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

### Tools
In *LangChain*, a **tool** is a callable interface (*typically a Python function*) that acts as a specialized skill for an AI agent, allowing it to interact with the outside world, execute code, fetch real-time data, or call APIs. Tools enable LLMs to move beyond text generation by providing structured inputs and receiving outputs to perform actions.

Tools allow LLMs to:
- fetch real-time data
- query databases
- perform calculations
- trigger business logic

> *The LLM decides **when** and **how** to use the tool.*

#### Conceptual Tool Example
```python
# conceptual_tool_example.py
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
- the LLM can call get_country_population ("Slovakia")
- receive the result
- use it to generate a final response

This is how LLMs move from *text-only* to **action-capable systems**.

### Memory
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

> *Below examples of manual memory strategies help clarify how memory works conceptually, before introducing framework-provided memory abstractions such as those in LangChain.*

#### Manual Short-Term Memory Example
```python
# manual_shortterm_memory.py
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
```python
# manual_longterm_memory.py
import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Get the directory where the script itself is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Combine that directory with the filename
file_path = os.path.join(script_dir, "history.json")

# Load .env
load_dotenv()

# Initialize model
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.2
)

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

- initial content of the file `history.json`
```json
[
    {
        "role": "human",
        "content": "My name is Daniel and I love biking."
    }
]
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

#### Framework-Provided Memory
In professional AI development, short-term memory isn't just a growing list - it’s a managed state. LangChain provides set of tools, which allow to build professional state management system.

Below example displays the LangGraph Checkpointer approach. It treats memory as a *"database of states"*.

```python
# langchain_shortterm_memory.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# Load .env
load_dotenv()

# Initialize model
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.2
)

# Create the Agent
agent = create_agent(
    model = llm,
    tools = [],
    system_prompt = "You are a helpful assistant.",
    checkpointer = InMemorySaver()
)

while True:
    # Get user input
    user_input = input("You: ").strip()
    if user_input.lower() in ("exit", "quit", "q"):
        print("Bye!")
        break

    # Invoke LLM
    ai_message = agent.invoke(
        {
            "messages": [
                HumanMessage(content = user_input)
            ]
        },
        {
            "configurable": {
                "thread_id": "1"
            }
        }
    )

    # Get response content
    content = ai_message["messages"][-1].content
    print("Assistant:", content)

```

With short-term memory enabled, long conversations can exceed the LLM's context window. Next sections are going to introduce message **trimming** and **summarization**.

#### Memory Growth and Trim Messages
Trim Messages provides strategy of maximum tokens allowed in the message history and strategy to be applied for handling the boundary. The messages are truncated once the limit is reached.

```python
# langchain_shortterm_memory_trim.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from typing import Any

# Load OPEN_AI_KEY
load_dotenv()

# Define LLM
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0
)

# --- TRIMMING LOGIC ---
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    Trim conversation history to keep the context window bounded.
    - Keeps only the last MAX_MESSAGES messages.
    - Returns None if no trimming is required.
    """
    MAX_MESSAGES = 4

    messages = state["messages"]

    # Nothing to trim
    if len(messages) <= MAX_MESSAGES:
        return None  # No changes needed

    recent_messages = messages[-MAX_MESSAGES:]
    new_messages = recent_messages

    return {
        "messages": [
            RemoveMessage(id = REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

# Create the Agent with the trimming logic as a state_modifier
agent = create_agent(
    model = llm,
    tools = [],
    system_prompt = "You are a helpful assistant.",
    middleware = [trim_messages],
    checkpointer = InMemorySaver()
)

while True:
    # Get user input
    user_input = input("You: ").strip()
    if user_input.lower() in ("exit", "quit", "q"):
        print("Bye!")
        break

    # Invoke LLM
    ai_message = agent.invoke(
        {
            "messages": [
                HumanMessage(content = user_input)
            ]
        },
        {
            "configurable": {
                "thread_id": "1"
            }
        }
    )

    # Get response content
    content = ai_message["messages"][-1].content
    print("Assistant:", content)
```

#### Memory Growth and Summarization
As memory grows over time, storing the full conversation history may become inefficient or impractical. Moreover, trimming may lose important messages.

A common optimization strategy is **memory summarization**, where:
- older conversation segments are periodically summarized
- detailed message history is replaced with concise semantic summaries
- the overall context is preserved while significantly reducing storage size

This approach allows memory to scale while remaining performant and cost-efficient.

```python
# langchain_shortterm_memory_summarize.py
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# Load .env
load_dotenv()

# Initialize model
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.2
)

# Create the Agent
agent = create_agent(
    model = llm,
    tools = [],
    system_prompt = "You are a helpful assistant.",
    middleware = [
        SummarizationMiddleware(
            model = llm,
            trigger = ("tokens", 4000),
            keep = ("messages", 20)
        )
    ],
    checkpointer = InMemorySaver()
)

while True:
    # Get user input
    user_input = input("You: ").strip()
    if user_input.lower() in ("exit", "quit", "q"):
        print("Bye!")
        break

    # Invoke LLM
    ai_message = agent.invoke(
        {
            "messages": [
                HumanMessage(content = user_input)
            ]
        },
        {
            "configurable": {
                "thread_id": "1"
            }
        }
    )

    # Get response content
    content = ai_message["messages"][-1].content
    print("Assistant:", content)

```

- `trigger = ("tokens", 4000)` - when total token count exceeds 4000 → summarize
- `keep = ("messages", 20)` - after summarization → keep last 20 raw messages

#### Memory Strategy Comparison Table
| Strategy      | Preserves Full Context | Token Efficient | Production Suitable | Risk             |
| ------------- | ---------------------- | --------------- | ------------------- | ---------------- |
| Manual List   | Yes                    | ❌ No           | ❌ No              | Context overflow |
| Trim          | Partial                | ✅ Yes          | ⚠️ Depends         | Information loss |
| Summarization | Semantic               | ✅ Yes          | ✅ Yes             | Summary drift    |

> In production deployments, a persistent checkpointer (*e.g., SQLite or Postgres*) should be used instead of `InMemorySaver` to ensure durability across restarts.

## Retrieval-Augmented Generation (RAG)
**Retrieval-Augmented Generation (RAG)** is an AI framework that combines information retrieval with generative AI to create more accurate, up-to-date, and contextually relevant responses. It works by first retrieving relevant information from an external knowledge base, like a company's internal documents or the latest internet data, and then feeding this information into a Large Language Model (LLM) to generate a response. This process grounds the LLM's answer in specific facts, reduces the risk of "hallucinations" (*generating incorrect information*), and allows the model to reference sources, making it more reliable and efficient than simply relying on its pre-trained knowledge.

**LlamaIndex** excels in search and retrieval tasks. It’s a powerful tool for data indexing and querying and a great choice for projects that require advanced search. LlamaIndex enables the handling of large datasets, resulting in quick and accurate information retrieval.

### Minimal RAG Example
```python
# rag_basic.py
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# Get the directory where the script itself is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Combine that directory with the folder name
file_path = os.path.join(script_dir, "data")

# Load .env
load_dotenv()

# Initialize model
llm = OpenAI(
    model="gpt-4o-mini",
    temperature = 0.2
)

# Load documents from disk
documents = SimpleDirectoryReader(file_path).load_data()

# Build vector index
index = VectorStoreIndex.from_documents(
    documents,
    llm=llm,
)

# Create query engine
query_engine = index.as_query_engine()

# Ask a question
response = query_engine.query(
    "How many days per week can employees work remotely?"
)

# Print model output
print(response)

```

- content of the file `data/company_policy.txt`
```text
Employees are allowed to work remotely up to three days per week.
All security incidents must be reported within 24 hours.
```

What’s happening internally:
- documents are split into chunks
- chunks are embedded
- relevant chunks are retrieved
- retrieved context is injected into the prompt
- LLM generates a grounded response

> **NOTE:**
>
> *LlamaIndex and LLM SDKs may emit HTTP and embedding logs by default. These can be safely silenced by adjusting Python logging levels without  affecting functionality or results.*


### Explicit Retrieval + Generation Pipeline Example
```python
# rag_explicit.py
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# Get the directory where the script itself is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Combine that directory with the filename
file_path = os.path.join(script_dir, "data")

# Load .env
load_dotenv()

# Initialize model
llm = OpenAI(
    model="gpt-4o-mini",
    temperature = 0.2
)

# Load documents from disk
documents = SimpleDirectoryReader(file_path).load_data()

# Build vector index
index = VectorStoreIndex.from_documents(documents)

retriever = index.as_retriever(similarity_top_k = 2)

nodes = retriever.retrieve(
    "What must be reported within 24 hours?"
)

print("Retrieved context:")
for node in nodes:
    print("-", node.text)

context = "\n".join(node.text for node in nodes)

prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
What must be reported within 24 hours?
"""

response = llm.complete(prompt)
print("\nFinal answer:")
print(response.text)

```

The above example:
- allow to see what the model sees
- provide easier way to debug hallucinations
- serve as a preparation example for custom RAG pipelines

## Document Indexing and Querying
**Document indexing and querying** form the backbone of Retrieval-Augmented Generation (RAG) systems. This process defines how raw documents are transformed into searchable knowledge and how relevant information is later retrieved and injected into Large Language Model (LLM) prompts.

**Indexing** typically involves loading documents, splitting them into meaningful chunks, generating vector embeddings, and storing those embeddings in an index optimized for similarity search. **Querying** then uses this index to efficiently identify the most relevant pieces of information in response to a user’s question.

**The quality of indexing directly affects retrieval accuracy.** Poor chunking, insufficient metadata, or inappropriate index configurations can result in missing context or irrelevant results — issues that cannot be fully compensated for by prompt engineering or more powerful models. Conversely, well-designed indexing enables LLMs to generate grounded, factual, and context-aware responses.

In this section, we explore how *LangChain* and *LlamaIndex* handle document indexing and querying, focusing on practical design choices, common trade-offs, and techniques for building reliable and scalable retrieval pipelines.

### Document Loading
**Document loading** is the first step in the document indexing pipeline. It is the process of ingesting raw data from various sources and converting it into a standardized internal representation that can be further processed, chunked, embedded, and indexed.

The quality and structure of loaded documents directly influence all downstream stages of a Retrieval-Augmented Generation (RAG) system. Poorly loaded or incomplete data can result in missing context, inaccurate retrieval, or misleading responses, regardless of how well chunking or querying is implemented.

*LlamaIndex* provides a flexible document loading system that supports multiple data sources, formats, and preprocessing strategies, allowing developers to adapt ingestion pipelines to different use cases.

#### Supported Sources
Documents can be loaded from a variety of sources, including:
- local text files
- PDFs and Word documents
- Markdown files
- web pages
- databases
- APIs and cloud storage

In this learning path, examples focus on local files to keep the core concepts clear and reproducible.

#### Loading Files
```python
# load_local_txt.py
import os
from llama_index.core import SimpleDirectoryReader

# Get the directory where the script itself is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Combine that directory with the filename
file_path = os.path.join(script_dir, "data")

# Load documents from disk
documents = SimpleDirectoryReader(file_path).load_data()

# Print output
print(f"Loaded {len(documents)} document(s).")

```

Each file becomes a `Document` object containing `text` and `metadata` (*filename, path, etc.*).

- draft of inspecting loaded documents
```python
doc = documents[0]

print("Text preview:")
print(doc.text[:300])

print("\nMetadata:")
print(doc.metadata)

```

#### Loading Specific File Types
```python
# load_specific_file.py
import os
from llama_index.core import SimpleDirectoryReader

# Get the directory where the script itself is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Combine that directory with the filename
file_path = os.path.join(script_dir, "data")

# Load documents from disk
documents = SimpleDirectoryReader(
    input_dir = "data",
    required_exts = [".txt", ".md"]
).load_data()

# Print output
print(f"Loaded {len(documents)} document(s).")

```

#### Recursive Loading
```python
import os
from llama_index.core import SimpleDirectoryReader

# Get the directory where the script itself is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Combine that directory with the filename
file_path = os.path.join(script_dir, "data")

# Load documents from disk
documents = SimpleDirectoryReader(
    input_dir = "data",
    recursive = True
).load_data()

# Print output
print(f"Loaded {len(documents)} document(s).")

```

> **Recursive**: include the contents of any subdirectories.
>
> Is usefull for:
> - knowledge bases
> - documentation trees
> - internal wikis

#### Multiple Sources
```python
documents = []

documents += SimpleDirectoryReader("policies").load_data()
documents += SimpleDirectoryReader("manuals").load_data()
documents += SimpleDirectoryReader("reports").load_data()

```

#### Common Document Loading Pitfalls
- loading raw PDFs without text normalization
- including boilerplate content (headers, footers, navigation)
- mixing unrelated document types in a single index
- ignoring metadata during ingestion

### Metadata Enrichment
*Metadata* provides additional context about documents, such as file names, timestamps, categories, or access levels. This metadata can later be used for filtering, ranking, or auditing retrieved results.

> Think of **metadata** as *labels attached to chunks of knowledge*. Metadata answers where it came from, what it is, and how it should be used.

For example:
> “Employees can work remotely up to three days per week.”

*Without* metadata, this sentence is just text. *With* metadata, it becomes:
- this comes from internal docs
- specifically from `company_policy.txt`
- potentially: HR-related, confidential, policy, versioned, etc.

Therefore, on the base of metadata you can:
- filter results
- trace answers back to sources
- apply access control
- debug why something was retrieved

```python
# metadata_example.py
import os
from llama_index.core import SimpleDirectoryReader

# Get the directory where the script itself is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Combine that directory with the filename
file_path = os.path.join(script_dir, "data")

# Load documents from disk and define metadata
documents = SimpleDirectoryReader(
    input_dir = file_path,
    file_metadata = lambda filename: {
        "source": "internal_docs",
        "document_type": "policy",
        "filename": os.path.basename(filename),
    }
).load_data()

# Inspect loaded documents
doc = documents[0]

print("Document text preview:")
print(doc.text[:200])

print("\nDocument metadata:")
print(doc.metadata)

```

> **Metadata enrichment** enables filtering, traceability, and governance in RAG systems.
> While optional for small demos, metadata becomes essential in production-scale applications.

### Document Chunking Strategies
**Document chunking** is the process of splitting large documents into smaller, manageable pieces (*chunks*) before they are indexed and embedded. Since Large Language Models (LLMs) have context length limits and operate on token-based inputs, chunking plays a critical role in ensuring that relevant information can be efficiently retrieved and accurately used during response generation.

Effective chunking improves retrieval accuracy, reduces hallucinations, and helps balance performance, cost, and contextual completeness in **Retrieval-Augmented Generation (RAG)** systems.

#### Why Chunking Matters
- LLMs cannot process entire large documents at once
- retrieval operates on chunks, not whole files
- poor chunking leads to:
  - missing context
  - irrelevant retrievals
  - increased hallucinations

#### Fixed-Size Chunking
**Fixed-size chunking** is a straightforward, computationally efficient RAG technique that splits text into uniform segments based on a set number of characters, tokens, or words, typically using a 10-20% overlap to maintain context. It is ideal for quick prototyping and uniform text, though it risks breaking sentences and semantic coherence (*split sentences or ideas, context may be fragmented*).

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

documents = SimpleDirectoryReader("data").load_data()

splitter = SentenceSplitter(
    chunk_size = 512,
    chunk_overlap = 50,
)

index = VectorStoreIndex.from_documents(
    documents,
    transformations = [splitter]
)

```

> Good default values:
> - `chunk_size`: 256–1024 tokens
> - `chunk_overlap`: 10–20%

#### Sentence-Based Chunking
**Sentence-based chunking** is a natural language processing strategy that splits text at sentence boundaries (*e.g., periods, question marks*) to create chunks containing one or more full sentences. This method ensures high semantic coherence and readability for RAG and Q&A systems, as it avoids cutting off mid-thought unlike fixed-size methods.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size = 400,
    chunk_overlap = 50,
    separator = " ",
)

index = VectorStoreIndex.from_documents(
    documents,
    transformations = [splitter]
)

```

> This is the **recommended default** for most RAG use cases.

#### Paragraph and Section-Based Chunking
**Paragraph and section-based chunking** are structural strategies in Retrieval-Augmented Generation (RAG) that divide documents based on natural formatting — paragraphs, headers, or markdown — to preserve semantic context. These methods create coherent, logical chunks ideal for structured documents like manuals or reports, enhancing retrieval relevance compared to fixed-size methods.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size = 800,
    chunk_overlap = 100,
)

index = VectorStoreIndex.from_documents(
    documents,
    transformations = [splitter]
)

```

#### Sliding Window Chunking
**Sliding window chunking** is a text segmentation strategy for RAG that creates overlapping text segments by moving a fixed-size window over data, usually with a 10–20% overlap, to preserve context between chunks. It prevents information loss at boundaries, enhancing retrieval accuracy for continuous text like legal or medical documents, though it increases storage costs and redundancy.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size = 300,
    chunk_overlap = 150,
)

index = VectorStoreIndex.from_documents(
    documents,
    transformations = [splitter]
)

```

#### Inspecting Chunks
The below code allows to see what model sees and helps identify why retrieval succeeds or fails.

```python
nodes = splitter.get_nodes_from_documents(documents)

for i, node in enumerate(nodes[:3]):
    print(f"\n--- Chunk { i + 1 } ---")
    print(node.text)

```

#### Choosing a Strategy

| Use Case           | Strategy                 |
| ------------------ | ------------------------ |
| Simple docs        | Fixed-size               |
| Q&A systems        | Sentence-based           |
| Manuals / policies | Section-based            |
| Legal / long text  | Sliding window           |
| Production RAG     | Sentence-based + overlap |

> ***Chunking** is not a preprocessing detail — it is a core design decision in RAG systems. Poor chunking cannot be fixed by better prompts or stronger models.*

### Embeddings and Index Construction

#### What Embeddings Represent
**Embeddings** are dense numerical vectors that capture the semantic meaning of text. Instead of representing words as discrete symbols, embeddings position text in a high-dimensional space where:
- semantically similar texts are close together
- unrelated texts are far apart
- meaning is captured beyond exact keyword matching

In RAG systems, embeddings are the bridge between unstructured text and efficient retrieval.

> **Key idea:** Retrieval happens in *vector space*, not text space.

Example:
```text
"Remote work policy"
"Working from home rules"
```

Different words, but embeddings place them close together.

#### How Chunking Affects Embeddings
LLMs embed chunks, not entire documents. The chunking strategy directly shapes what the embeddings represent.

| Chunking Choice       | Effect on Embeddings                  |
| --------------------- | ------------------------------------- |
| Very large chunks     | Embeddings become vague and unfocused |
| Very small chunks     | Embeddings lose context               |
| Sentence-based chunks | High precision, lower context         |
| Section-based chunks  | Balanced meaning and context          |
| Overlapping chunks    | Better recall, higher cost            |

```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size = 512,
    chunk_overlap = 50,
)

nodes = splitter.get_nodes_from_documents(documents)

```

#### Building an Index from Documents
Once documents are:
- loaded
- chunked
- embedded

They can be stored in an index for retrieval. The most common index for RAG is the **Vector Store Index**.

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)

```

#### Common embedding pitfalls
- chunking too large, which leads to:
  - embedding mixes multiple topics
  - retrieval becomes inaccurate
- chunking too small, which leads to:
  - embeddings lack context
  - model retrieves irrelevant fragments
- ignoring metadata
  - cannot filter by source
  - cannot scope queries
  - debugging retrieval becomes painful
- re-embedding documents
  - cost time
  - cost money
  - introduces inconsistencies
- embeddings:
  - do not reason
  - do not answer questions
  - only retrieve relevant context

> chunking defines meaning → embeddings encode meaning → indexes enable retrieval → LLM generates answers

### Querying and Retrieval
Once documents are indexed, the system’s main job becomes retrieval: finding the most relevant chunks to answer a user’s question.
Retrieval is driven by semantic similarity, not exact keyword matching.

#### Basic Semantic Querying 
**Basic semantic querying** is the simplest form of interaction with a vector index.
- user submits a question
- the question is converted into an embedding
- the index finds the most similar document chunks
- retrieved chunks are passed to the LLM
- the LLM generates a grounded answer

```python
# Define query (no metadata filtering)
query_engine = index.as_query_engine()

# Query -> Response
response = query_engine.query(
    "What is the remote work policy?"
)

# Print output
print(f"Response: {response}\n")

```

> **Key characteristic:** The user does not need to match document wording exactly.

#### Semantic Search
Semantic search focuses on *finding relevant information*, not necessarily generating a long answer.

**Semantic search vs basic querying**
| Feature  | Basic Querying          | Semantic Search |
| -------- | ----------------------- | --------------- |
| Output   | Natural language answer | Relevant chunks |
| Uses LLM | Yes                     | Optional        |
| Best for | Q&A                     | Discovery       |
| Cost     | Higher                  | Lower           |


```python
# Create retriever
retriever = index.as_retriever(similarity_top_k = 2)

# Retrieving nodes
nodes = retriever.retrieve("remote work policy")
for node in nodes:
    print(f"Text:\n{node.text}\n")
    print(f"Metadata: {node.metadata}")

```

#### Metadata-Base Filtering
```python
# metadata_access_control.py
import logging
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

# Set logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)

# Load .env
load_dotenv()

# Get the directory where the script itself is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Combine that directory with the filename
file_path = os.path.join(script_dir, "data")

# Load documents from disk and define metadata
documents = SimpleDirectoryReader(
    input_dir = file_path,
    file_metadata = lambda filename: {
        "source": "internal_docs",
        "document_type": "policy",
        "filename": os.path.basename(filename),
    }
).load_data()

# Build vector index
index = VectorStoreIndex.from_documents(documents)

# Define metadata filters explicitly
filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key = "source",
            value = "internal_docs"
        ),
        MetadataFilter(
            key = "document_type",
            value = "policy"
        )
    ],
    condition = "and"  # Combine filters with AND (default)
)

# Metadata filtering during quering
query_engine = index.as_query_engine(filters = filters)

# Query -> Response
response = query_engine.query(
    "How many days per week is remote work allowed?"
)

# Print output
print(f"Response: {response}\n")

# Inspect what was retrieved
for node in response.source_nodes:
    print(f"Text:\n{node.text}\n")
    print(f"Metadata: {node.metadata}")

```

- draft of example with `OR` condition
```python
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

# Define metadata filters explicitly
filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key = "document_type",
            value = "policy"
        ),
        MetadataFilter(
            key = "document_type",
            value = "guideline"
        )
    ],
    condition = "or"
)

# Metadata filtering during quering
query_engine = index.as_query_engine(filters=filters)

# Query -> Response
response = query_engine.query(
    "How many days per week is remote work allowed?"
)

```

#### Debugging Retrieval
```python
# metadata_retrieving.py
import logging
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Set logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)

# Load .env
load_dotenv()

# Get the directory where the script itself is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Combine that directory with the filename
file_path = os.path.join(script_dir, "data")

# Load documents from disk and define metadata
documents = SimpleDirectoryReader(
    input_dir = file_path,
    file_metadata = lambda filename: {
        "source": "internal_docs",
        "document_type": "policy",
        "filename": os.path.basename(filename),
    }
).load_data()

# Build vector index
index = VectorStoreIndex.from_documents(documents)

# Create retriever
retriever = index.as_retriever(similarity_top_k = 2)

# Retrieving nodes
nodes = retriever.retrieve("remote work policy")
for node in nodes:
    print(f"Text:\n{node.text}\n")
    print(f"Metadata: {node.metadata}")

```

### Combining Metadata and Chunking
Combining *metadata* with *chunking strategies* is essential for building production-ready Retrieval-Augmented Generation (RAG) systems. While **chunking** breaks large documents into manageable, semantic pieces (*e.g., sentences, paragraphs*), **metadata** provides the *"context about the content"* — such as document title, page number, section headers, author, or data type (*e.g., code vs. text*).

By tagging every chunk with structured metadata, you can filter, prioritize, and manage data efficiently, improving retrieval precision and preventing *"cross-document contamination"*.

```text
Raw Document
   ↓ (Document Loading + Metadata Enrichment)
Document + Metadata
   ↓ (Chunking)
Chunks + Inherited Metadata
   ↓ (Embedding & Indexing)
Vector Index
   ↓ (Metadata Filtering + Similarity Search)
Retrieved Chunks
   ↓
LLM Answer
```
> Metadata is attached before chunking, but applied after chunking.

#### Load Document with Metadata
```python
# metadata_and_chunking.py
import os
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_dir = "data",
    file_metadata = lambda filename: {
        "source": "internal_docs",
        "document_type": "policy",
        "filename": os.path.basename(filename),
    },
).load_data()

```

- metadata is attached to the document
- no chunking yet

#### Apply a Chunking Strategy (Sentence-Based)
```python
# metadata_and_chunking.py
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex

splitter = SentenceSplitter(
    chunk_size = 300,
    chunk_overlap = 50,
)

index = VectorStoreIndex.from_documents(
    documents,
    transformations = [splitter],
)

```

- the document is spread into chunks
- metadata is copied to every chunk derived from the document

#### Inspect Chunks + Metadata
```python
# metadata_and_chunking.py
nodes = splitter.get_nodes_from_documents(documents)

for node in nodes:
    print(f"Chunk text:\n{node.text}\n")
    print(f"Metadata: {node.metadata}")

``` 

- inspect what the retriever actually works with
- inspect whether chunk sizes make sense
- inspect whether metadata survived intact

#### Query Using Metadata Filtering
```python
# metadata_and_chunking.py
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

filters = MetadataFilters(
    filters = [
        MetadataFilter(
            key = "document_type",
            value = "policy"
        ),
        MetadataFilter(
            key = "source",
            value = "internal_docs"
        )
    ],
    condition = "and"
)

query_engine = index.as_query_engine(filters = filters)

response = query_engine.query(
    "How many days per week can employees work remotely?"
)

print(response)

```

- only policy chunks are considered
- chunking ensures semantic precision
- metadata ensures domain correctness

### Common Indexing Pitfalls
- applying metadata after indexing (too late)
- using inconsistent metadata keys
- chunk sizes that don’t match question granularity
- overlapping metadata filters that return zero chunks

## Modular AI Pipelines
As AI applications grow in complexity, a single prompt or script quickly becomes insufficient. **Modular AI pipelines** address this by breaking AI systems into composable, interchangeable components that can be developed, tested, and evolved independently.

Both **LangChain** and **LlamaIndex** are designed around this principle.

### Composable Components
A modular AI pipeline typically consists of the following building blocks:
- Data Ingestion
  - document loaders
  - metadata enrichment
  - chunking strategies
- Indexing & Retrieval
  - vector indexes
  - retrievers (semantic, hybrid, filtered)
- Reasoning & Generation
  - LLMs
  - prompt templates
  - response synthesizers
- Memory & State
  - short-term conversation context
  - long-term memory stores
- Tools & Integrations
  - APIs
  - databases
  - search engines

Each component has a single responsibility, which makes the system easier to maintain and extend.

### Swapping Retrievers, LLMs, Tools
One of the biggest advantages of modern AI frameworks is the ability to swap components without rewriting the pipeline.

#### Swapping Retrievers
```python
# Default semantic retriever
retriever = index.as_retriever(similarity_top_k = 3)

# Later: metadata-filtered retriever
retriever = index.as_retriever(
    similarity_top_k = 5,
    filters = filters
)

```

The rest of the pipeline remains unchanged.

#### Swapping LLMs
```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(model = "gpt-4o-mini")

query_engine = index.as_query_engine(llm = llm)

```

Later, switching models:
```python
llm = OpenAI(model = "gpt-4.1")

query_engine = index.as_query_engine(llm = llm)

```

No changes to:
- index
- chunking
- metadata
- retrieval logic

#### Adding Tools to the Pipeline
```python
@tools
def fetch_employee_count():
    return "The company has 120 employees."

# Tool-enabled reasoning layer

```

Tools can be:
- added
- removed
- mocked
- replaced

…without impacting retrieval or indexing.

### Production-Ready Patterns

#### Clear separation of concerns
- retrieval logic ≠ generation logic
- data loading ≠ indexing
- prompt design ≠ business logic

#### Configuration Over Code
```python
PIPELINE_CONFIG = {
    "model": "gpt-4o-mini",
    "top_k": 4,
    "chunk_size": 512,
}

```

Avoid hardcoding decisions inside functions.

#### Observability and Logging
Track:
- retrieved chunks
- token usage
- latency
- model errors

This is critical for debugging and cost control.

#### Fallbacks and Guardrails
- retry on API failures
- use smaller models for low-risk queries
- validate structured outputs
- apply metadata filters defensively

#### Stateless Core + Optional Memory
Design pipelines so they:
- work without memory
- improve with memory

This allows safe scaling and easier testing.

> **Mental Model**
>
> A modular AI pipeline is not a single “AI system” — it is a flexible assembly of specialized parts.

## Summary
This chapter explored how **LangChain** and **LlamaIndex** enable the construction of structured, reliable, and scalable AI applications beyond simple prompt–response interactions.

We started by introducing chains, tools, and memory, establishing how complex behaviors can be decomposed into manageable, reusable components. By separating reasoning, action, and state, AI systems become easier to extend, debug, and maintain.

We then examined Retrieval-Augmented Generation (RAG) as a foundational pattern for grounding model outputs in external knowledge. Through practical examples, we demonstrated how retrieval, chunking, metadata enrichment, and filtering work together to reduce hallucinations and improve factual accuracy.

The document indexing and querying section showed how raw data is transformed into searchable knowledge. We covered document loading, chunking strategies, embeddings, index construction, and semantic retrieval—highlighting how each design choice impacts accuracy, performance, and cost.

Finally, we introduced modular AI pipelines, emphasizing composability, interchangeability, and production-readiness. By treating retrievers, LLMs, tools, and memory as swappable components, developers can evolve AI systems safely and incrementally while maintaining clarity and control.

Together, these concepts form a strong foundation for building AI-powered systems that reason over data, adapt to context, and scale beyond experimentation.

## What’s Next
The next section, **Automation & Agent-Based Systems**, builds on this foundation by shifting from single, modular pipelines to autonomous and semi-autonomous agents.

This chapter will focus on:
- single-agent vs multi-agent system design
- CrewAI and AutoGen fundamentals
- task orchestration, role specialization, and delegation
- failure handling, retries, and observability
