# Example: AI Assistant Using Python Libraries LlamaIndex and LangChain

## Table of content
- [Objective](#objective)
- [Legal & Safety Disclaimer](#legal--safety-disclaimer)
- [Demonstration & Evaluation](#demonstration--evaluation)
- [Success Criteria & KPIs](#success-criteria--kpis)
- [Development Requirements](#development-requirements)
- [Development Steps](#development-steps)
- [Future Improvements](#future-improvements)

## Objective
This project serves as a presentation and simulation demonstrating how AI can be used in a business environment for troubleshooting purposes.  
The simulation topic focuses on a set of well-known medical conditions (*e.g. flu, cough, burns, headache, etc.*).

It is assumed that users are already familiar with these common conditions, including their typical symptoms and general treatment approaches.

## Legal & Safety Disclaimer
This project is intended **solely for educational, demonstration, and simulation purposes**.

The AI assistant does **not** provide medical advice, medical diagnosis, or treatment recommendations.  
All information presented by the assistant is based on predefined, high-level example data and is designed to demonstrate AI troubleshooting workflows rather than real-world medical decision-making.

Users must not rely on this system for health-related decisions.  
For any medical concerns, users should always consult a qualified healthcare professional.

The project intentionally limits medical detail, avoids personalized medical recommendations, and focuses on illustrating AI concepts such as Retrieval-Augmented Generation (RAG), prompt design, and output validation.

> *“This AI assistant is for demonstration purposes only and does not provide medical advice.”*

## Demonstration & Evaluation
The user interacts with the AI assistant by describing observed symptoms.  
Based on the provided symptoms, the AI assistant should:
- identify the most relevant condition using the available knowledge base
- provide a high-level treatment or resolution suggestion

The focus is on demonstrating reasoning, retrieval accuracy, and response quality rather than medical diagnosis.

## Success Criteria & KPIs
The success of the AI assistant is evaluated based on functional accuracy, response quality, and system behavior within defined safety boundaries.

### Functional Accuracy
- **Condition Recognition Accuracy**
  - the assistant correctly identifies the relevant condition based on user-provided symptoms
  - neasured by comparison against known examples from `common_health_issues.txt`
- **Relevant Knowledge Retrieval**
  - retrieved document chunks are directly related to the user query
  - minimal retrieval of unrelated or redundant context

### Response Quality
- **Clarity and Readability**
  - responses are easy to understand for non-technical users
  - medical terminology is avoided or explained at a high level
- **Structure Consistency**
  - responses follow a predictable structure (*e.g. description, symptoms, solution*)
  - output aligns with the structure of the source documents
- **Completeness**
  - the response addresses the user’s symptoms and provides a general resolution approach
  - no critical information gaps in the generated answer

### RAG Performance
- **Chunking Effectiveness**
  - different chunk sizes and overlap strategies are evaluated
  - performance is measured by retrieval relevance and answer consistency
- **Metadata Utilization**
  - metadata (*e.g. issue type, severity, keywords*) improves retrieval precision
  - demonstrable impact compared to non-metadata-based retrieval

### Safety and Boundary Compliance
- **Disclaimer Enforcement**
  - the assistant consistently includes or respects the legal and safety disclaimer
  - no medical diagnosis or personalized treatment is generated
- **Refusal Handling**
  - the assistant appropriately declines requests for medical advice, prescriptions, or diagnoses
  - refusals are polite, clear, and redirect the user to safe alternatives

### System Reliability
- **Response Consistency**
  - similar inputs produce logically consistent outputs
  - no hallucinated conditions outside the provided knowledge base
- **Latency**
  - responses are generated within an acceptable time frame for interactive demos
  - target latency defined based on local vs. remote LLM setup

## Development Requirements
Leverage knowledge from:
- `Chapter 1: Python and AI Integration`
- `Chapter 2: LangChain and LlamaIndex`

Key implementation requirements:
- use a local knowledge source: `common_health_issues.txt` with predefined structured data
- design clear and specific prompts
- apply Retrieval-Augmented Generation (RAG), metadata enrichment, and related techniques
- validate, monitor, and evaluate LLM outputs for correctness and relevance

Experiment with different document chunking strategies and compare their impact on retrieval accuracy and response quality.

## Development Steps
First, the documents are loaded into the system. In this implementation, a single structured text file is used as the source of knowledge. The document is enriched with metadata attributes, which are subsequently employed for document-level filtering during retrieval.
```python
# ai_assistant_v1.py

# Get the directory where the script itself is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Combine that directory with the filename
file_path = os.path.join(script_dir, "data", "common_health_issues.txt")

# Load documents from disk and define metadata
documents = SimpleDirectoryReader(
    input_files = [file_path],
    file_metadata = lambda filename: {
        "source": "internal_docs",
        "domain": "health_simulation",
        "document_type": "medical_examples",
        "audience": "demo",
        "risk_level": "low",
        "content_scope": "educational_only",
        "filename": os.path.basename(filename),
    }
).load_data()
```

Chunking refers to the process of partitioning large documents into smaller, semantically coherent units (*chunks*) prior to indexing and embedding. The selection of an appropriate chunking strategy depends on the source file format, internal structure, and overall document size. As an initial configuration, a simple text file is processed using a **Section-Aware Chunking** strategy.
```python
# ai_assistant_v1.py

# Define chunking strategy (Section-Aware Chunking)
splitter = SentenceSplitter(
    chunk_size = 600,
    chunk_overlap = 100,
)
```

A vector-based index is then constructed over the document using the defined chunking strategy, enabling efficient semantic retrieval.
```python
# ai_assistant_v1.py

# Build vector index + apply chunking strategy
index = VectorStoreIndex.from_documents(
    documents,
    transformations = [splitter]
)
```

Metadata filtering is applied during retrieval to ensure that only nodes satisfying the specified constraints are selected. These nodes are embedded into the prompt and made available for inspection via `response.source_nodes`. Consequently, the retrieval process is strictly limited to the designated document(s).
```python
# ai_assistant_v1.py

# Define metadata filters explicitly
filters = MetadataFilters(
    filters = [
        MetadataFilter(
            key = "domain",
            value = "health_simulation"
        ),
        MetadataFilter(
            key = "document_type",
            value = "medical_examples"
        ),
        MetadataFilter(
            key = "content_scope",
            value = "educational_only"
        )
    ],
    condition = "and"
)
```

Query processing and retrieval aim to identify the most relevant document chunks required to answer a user’s query. This process is governed by semantic similarity rather than exact keyword matching. In this context, only nodes originating from documents that satisfy the metadata filtering criteria are considered during retrieval.
```python
# ai_assistant_v1.py

# Metadata filtering during quering
query_engine = index.as_query_engine(
    filters = filters,
    similarity_top_k = 3
```

It is important to note that metadata filtering alone does not prevent the language model from leveraging its pre-trained knowledge. To ensure that responses are grounded exclusively in the provided document content, explicit prompt-level grounding must be employed.
```python
# ai_assistant_v1.py

from llama_index.core.prompts import PromptTemplate

text_qa_template_str = """You are an AI assistant for educational health simulations.
Use ONLY the information provided in the context below.
If the answer cannot be found in the context, say:
"I do not have enough information in the provided knowledge base."

Do NOT use external knowledge.
Do NOT provide medical diagnosis or advice.

Context:
{context_str}

Question:
{query_str}

Answer:
"""

# Metadata filtering during quering
query_engine = index.as_query_engine(
    filters = filters,
    similarity_top_k = 3,
    response_mode = "compact",
    text_qa_template = PromptTemplate(text_qa_template_str)
)
```

> **NOTE:**
>
> - `{query_str}` → comes from `.query("...")`
> - `{context_str}` → comes from retriever results
> - `PromptTemplate` → just defines where they go

Additionally, hallucination risk can be further mitigated by enforcing a minimum similarity threshold during retrieval. Document chunks with similarity scores below this threshold are excluded from further processing and are not passed to the language model.
```python
# ai_assistant_v1.py

from llama_index.core.postprocessor import SimilarityPostprocessor

query_engine = index.as_query_engine(
    filters=filters,
    similarity_top_k = 3,
    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff = 0.75)
    ]
)
```

Finally, the user’s query is converted into an embedding, the vector index identifies the most semantically similar document chunks, the retrieved chunks are supplied as contextual input to the language model, and a grounded response is generated.
```python
# Query -> Response
user_query = "I have a headache and sensitivity to light."
response = query_engine.query(user_query)

# Print model output
print("\n=== Final Answer ===")
print(response.response)
```

The following section is included to facilitate inspection and analysis of the retrieved document chunks that are provided to the language model.
```python
# Inspect source nodes
logger.info(f"=== Source Nodes ===")
for i, node_with_score in enumerate(response.source_nodes, start=1):
    node = node_with_score.node
    score = node_with_score.score

    logger.info(f"--- Source Node {i} ---")
    logger.info(f"Score: {score:.4f}")
    logger.info(f"Metadata: {node.metadata}")
    logger.info(f"Text:\n{node.text}")

# Inspect 'context_str'
context_str = "\n\n".join(
    n.node.text for n in response.source_nodes
)
logger.info("=== Context Sent to LLM ===")
logger.info(context_str)
```

## Future Improvements
Potential extensions to the project include:
- convert local knowledge source file `common_health_issues.txt` first into `markdown` and then into `yaml`
- build a side-by-side chunking comparison
- locally stored conversational memory between the user and the AI assistant
- design a troubleshooting conversation flow (symptom → clarification → suggestion)
- enabling the AI assistant to propose updates to `common_health_issues.txt` by adding new conditions while preserving the existing document structure
- tag issues with metadata (severity, urgency, keywords)
- build a “why this answer” explanation
- add test cases that intentionally fail

Next points:
- provide real-time user iteraction 
- how to: Success Criteria & KPIs
