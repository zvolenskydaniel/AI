# LangChain & LlamaIndex

## Overview
The previous chapter introduced the fundamentals of integrating Large Language Models (LLMs) into Python applications using direct API calls. While this approach is sufficient for simple use cases, real-world AI applications often require more structure, composability, and control.

This chapter introduces **LangChain** and **LlamaIndex**, two open-source frameworks designed to help build more complex LLM-powered systems. These frameworks provide higher-level abstractions for composing prompts, managing context and memory, integrating external tools and data sources, and orchestrating multi-step workflows.

Rather than replacing direct API usage, *LangChain* and *LlamaIndex* build on top of it, enabling scalable and maintainable AI application architectures.

## Goal
The goal of this chapter is to understand how higher-level frameworks can be used to design structured, modular, and production-oriented AI systems.

By the end of this chapter, the focus is on moving from single prompt–response interactions toward applications that combine multiple steps, external data, and reusable components.

## Core Concepts
- [Chains, Tools, and Memory](#chains-tools-and-memory)
- [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
- [Document Indexing and Querying](#document-indexing-and-querying)
- [Modular AI pipelines](#modular-ai-pipelines)
- [Why Frameworks Are Needed](#why-frameworks-are-needed)
- [Summary](#summary) 

## Chains, Tools, and Memory

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

#### Memory Growth and Summarization

As long-term memory grows over time, storing the full conversation history may become inefficient or impractical.

A common optimization strategy is **memory summarization**, where:
- older conversation segments are periodically summarized
- detailed message history is replaced with concise semantic summaries
- the overall context is preserved while significantly reducing storage size

This approach allows long-term memory to scale while remaining performant and cost-efficient.

> *These manual memory strategies help clarify how memory works conceptually, before introducing framework-provided memory abstractions such as those in LangChain.*

## Retrieval-Augmented Generation (RAG)
**Retrieval-augmented generation (RAG)** is an AI framework that combines information retrieval with generative AI to create more accurate, up-to-date, and contextually relevant responses. It works by first retrieving relevant information from an external knowledge base, like a company's internal documents or the latest internet data, and then feeding this information into a Large Language Model (LLM) to generate a response. This process grounds the LLM's answer in specific facts, reduces the risk of "hallucinations" (*generating incorrect information*), and allows the model to reference sources, making it more reliable and efficient than simply relying on its pre-trained knowledge.

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


### RAG with Explicit Retrieval + Generation Example
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

**The quality of indexing directly affects retrieval accuracy.** Poor chunking, insufficient metadata, or inappropriate index configurations can result in missing context or irrelevant results—issues that cannot be fully compensated for by prompt engineering or more powerful models. Conversely, well-designed indexing enables LLMs to generate grounded, factual, and context-aware responses.

In this section, we explore how *LangChain* and *LlamaIndex* handle document indexing and querying, focusing on practical design choices, common trade-offs, and techniques for building reliable and scalable retrieval pipelines.

### Document Loading
**Document loading** is the first step in the document indexing pipeline. It is the process of ingesting raw data from various sources and converting it into a standardized internal representation that can be further processed, chunked, embedded, and indexed.

The quality and structure of loaded documents directly influence all downstream stages of a Retrieval-Augmented Generation (RAG) system. Poorly loaded or incomplete data can result in missing context, inaccurate retrieval, or misleading responses, regardless of how well chunking or querying is implemented.

*LlamaIndex* provides a flexible document loading system that supports multiple data sources, formats, and preprocessing strategies, allowing developers to adapt ingestion pipelines to different use cases.

#### Supported Document Sources
Documents can be loaded from a variety of sources, including:
- local text files
- PDFs and Word documents
- Markdown files
- web pages
- databases
- APIs and cloud storage

In this learning path, examples focus on local files to keep the core concepts clear and reproducible.

#### Loading Local Text Files
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

#### Recursive Directory Loading
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
> Is usefull for:
> - knowledge bases
> - documentation trees
> - internal wikis

#### Metadata Enrichment
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

# Load documents from disk
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

- example how to debug why an answer was given (*nodes retrieving*)
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

# Load documents from disk
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
    print(f"Text: {node.text}\n")
    print(f"Metadata: {node.metadata}")

```

- draft of access control
```python
if node.metadata["document_type"] == "policy":
    # allow response

```

> Metadata enrichment enables filtering, traceability, and governance in RAG systems.
> While optional for small demos, metadata becomes essential in production-scale applications.

#### Loading Multiple Sources
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
    transformations = [splitter],
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
    transformations = [splitter],
)

```

> This is the **recommended default** for most RAG use cases.

#### Paragraph and Section-Based Chunking
**Paragraph and section-based chunking** are structural strategies in Retrieval-Augmented Generation (RAG) that divide documents based on natural formatting—paragraphs, headers, or markdown—to preserve semantic context. These methods create coherent, logical chunks ideal for structured documents like manuals or reports, enhancing retrieval relevance compared to fixed-size methods.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size = 800,
    chunk_overlap = 100,
)

index = VectorStoreIndex.from_documents(
    documents,
    transformations = [splitter],
)

```

#### Sliding Window Chunking
**Sliding window chunking** is a text segmentation strategy for RAG that creates overlapping text segments by moving a fixed-size window over data, usually with a 10–20% overlap, to preserve context between chunks. It prevents information loss at boundaries, enhancing retrieval accuracy for continuous text like legal or medical documents, though it increases storage costs and redundancy.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=300,
    chunk_overlap=150,
)

index = VectorStoreIndex.from_documents(
    documents,
    transformations=[splitter],
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

#### Choosing the Right Strategy

| Use Case           | Strategy                 |
| ------------------ | ------------------------ |
| Simple docs        | Fixed-size               |
| Q&A systems        | Sentence-based           |
| Manuals / policies | Section-based            |
| Legal / long text  | Sliding window           |
| Production RAG     | Sentence-based + overlap |

> ***Chunking** is not a preprocessing detail—it is a core design decision in RAG systems. Poor chunking cannot be fixed by better prompts or stronger models.*

### Embeddings and Index Construction
TBD

### Querying and Retrieval
TBD

### Common Indexing Pitfalls
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
