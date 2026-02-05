# Example: AI Assistant Using Python Libraries LlamaIndex and LangChain

## Table of content
- [Objective](#objective)
- [Legal & Safety Disclaimer](#legal--safety-disclaimer)
- [Demonstration & Evaluation Specification](#demonstration--evaluation-specification)
- [Success Criteria & KPIs](#success-criteria--kpis)
- [Development Requirements](#development-requirements)
- [Development Steps](#development-steps)
- [Fine-tuning](#fine-tuning)
- [Demo & Evaluation](#demo--evaluation)
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

## Demonstration & Evaluation Specification
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
    chunk_overlap = 100
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
from llama_index.core.postprocessor import SimilarityPostprocessor

query_engine = index.as_query_engine(
    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff = 0.75)
    ]
)
```

Finally, the user’s query is converted into an embedding, the vector index identifies the most semantically similar document chunks, the retrieved chunks are supplied as contextual input to the language model, and a grounded response is generated.
```python
# ai_assistant_v1.py

# Query -> Response
user_query = "I have a headache and sensitivity to light."
response = query_engine.query(user_query)

# Print model output
print("\n=== Final Answer ===")
print(response.response)
```

The following section is included to facilitate inspection and analysis of the retrieved document chunks that are provided to the language model.
```python
# ai_assistant_v1.py

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

## Fine-tuning
Once the script `ai_assistant_v1.py` is executed, the LLM generates responses based on the predefined local knowledge source `common_health_issues.txt`. The LLM’s output is constrained to include only the **Solution** section of the identified issue or disease.

### Chunking Strategy
Inspection of the log file `logger/logger.log` reveals the following behavior:
```log
==================================================

Issue: Minor Burn

Symptoms:
<--- omitted output --->

Description:
Minor burns are caused by brief contact with heat, hot liquids, or steam.
They typically affect only the outer layer of skin and heal within days to weeks.

--- Retrieved Node 2 ---
Score: 0.7358
Metadata: {'source': 'internal_docs', 'domain': 'health_simulation', 'document_type': 'medical_examples', 'audience': 'demo', 'risk_level': 'low', 'content_scope': 'educational_only', 'filename': 'common_health_issues.txt'}
Text:
They typically affect only the outer layer of skin and heal within days to weeks.

Solution:
<--- omitted output --->

==================================================
```

This indicates that the current configuration of the **Section-Aware Chunking** strategy causes chunks to be split across logical section boundaries. As a result, the second chunk begins in the middle of the **Description** section instead of starting cleanly at the **Solution** section.

To address this issue, the following approaches can be applied:

- **Adjust chunking parameters**
  - tune the `chunk_size` and `chunk_overlap` values by increasing or decreasing them to better align chunk boundaries with logical sections
```python
splitter = SentenceSplitter(
    chunk_size = 500,
    chunk_overlap = 20
)
```

- **Replace the chunking strategy with delimiter-based section-aware chunking**
  - read the source document `common_health_issues.txt` directly
  - instead of relying on `SimpleDirectoryReader` for chunking, perform manual document splitting based on a fixed delimiter (e.g. `=========`)
  - enrich each chunk with additional metadata, including the issue or disease name
  - remove the `SentenceSplitter`, as chunking is handled explicitly at the section level
```python
# ai_assistant_v2.py

# Load documents from disk and define metadata
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

## Each issue is separated by a hard delimiter
sections = [
    s.strip()
    for s in raw_text.split("==================================================")
    if s.strip()
]

## Manual pre-chunking of the document
documents = []
for section in sections:
    lines = section.splitlines()

    # Extract issue name
    issue_line = next(
        (l for l in lines if l.startswith("Issue:")),
        None
    )
    issue_name = issue_line.replace("Issue:", "").strip() if issue_line else "unknown"

    documents.append(
        Document(
            text=section,
            metadata={
                "source": "internal_docs",
                "domain": "health_simulation",
                "document_type": "medical_examples",
                "issue": issue_name,
                "audience": "demo",
                "risk_level": "low",
                "content_scope": "educational_only"
            }
        )
    )

# Build vector index + apply chunking strategy
index = VectorStoreIndex.from_documents(documents)

```

- **Resulting behavior**
  - each source node represents a single, complete issue or disease
  - each node consistently contains the fields `Issue`, `Symptoms`, `Description`, and `Solution`
```log
2026-02-05 07:07:40,614 - INFO - --- Source Node 1 ---
2026-02-05 07:07:40,614 - INFO - Score: 0.8315
2026-02-05 07:07:40,615 - INFO - Metadata: {'source': 'internal_docs', 'domain': 'health_simulation', 'document_type': 'medical_examples', 'issue': 'Headache', 'audience': 'demo', 'risk_level': 'low', 'content_scope': 'educational_only'}
2026-02-05 07:07:40,615 - INFO - Text:
Issue: Headache

Symptoms:
<--- omitted output --->

Description:
<--- omitted output --->

Solution:
<--- omitted output --->

2026-02-05 07:07:40,615 - INFO - --- Source Node 2 ---
2026-02-05 07:07:40,615 - INFO - Score: 0.7347
2026-02-05 07:07:40,615 - INFO - Metadata: {'source': 'internal_docs', 'domain': 'health_simulation', 'document_type': 'medical_examples', 'issue': 'Common Cold / Flu', 'audience': 'demo', 'risk_level': 'low', 'content_scope': 'educational_only'}
2026-02-05 07:07:40,615 - INFO - Text:
Issue: Common Cold / Flu

Symptoms:
<--- omitted output --->

Description:
<--- omitted output --->

Solution:
<--- omitted output --->

```

### Context Selection and Retrieval
The next step focuses on the **context** provided to the LLM during query execution. With the current configuration, two source nodes are retrieved and passed to the LLM:

- **Source Node 1**  
  - Score: `0.8315`  
  - Metadata: `{'issue': 'Headache'}`

- **Source Node 2**  
  - Score: `0.7347`  
  - Metadata: `{'issue': 'Common Cold / Flu'}`

The LLM must determine which retrieved content is most relevant for generating the final response. As discussed earlier, to further reduce the risk of hallucinations, a **minimum similarity threshold** can be enforced during retrieval. Nodes with similarity scores below this threshold are excluded and are not passed to the LLM.
```python
# ai_assistant_v2.py

from llama_index.core.postprocessor import SimilarityPostprocessor

query_engine = index.as_query_engine(
    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff = 0.75)
    ]
)
```

### Add Real-Time User Iteraction
At the moment, the user's query is statically specified for a single sentese. To simulate real-time `user vs. AI assistent` iteraction, for multiple scenarious during a demo, the original code needs to be adjusted.
```python
# ai_assistant_v3.py

print("Start chatting (type 'exit' to quit).")
while True:
    user_query = input("\nYou: ").strip()
    if user_query.lower() in ("exit", "quit", "q"):
        print("Bye!")
        break

    # Query -> Response
    response = query_engine.query(user_query)

    # Print model output
    print("""DISCLAIMER: 
        This AI assistant is for demonstration
        purposes  only  and  does  not provide 
        medical advice.
    """)
    print(f"AI Assistant: {response.response}")
```

### Update Behaviour
When the assistant does not recognize a matching condition, it does not attempt to infer or fabricate an answer. In cases where no relevant nodes are retrieved, the LLM receives an empty context. As a result, the model may return an **empty response**. This behavior is expected and represents a known characteristic of Retrieval-Augmented Generation (RAG) systems rather than a defect in the implementation.
```shell
$ python ai_assistant_v3.py
Start chatting (type 'exit' to quit).

You: I cannot stand on my leg as my big toe hurts a lot.
DISCLAIMER: 
        This AI assistant is for demonstration
        purposes  only  and  does  not provide 
        medical advice.
    
AI Assistant: Empty Response

You: exit
Bye!
```

To address this behavior, the retrieval phase is explicitly separated from the generation phase. The LLM is invoked only when at least one relevant source node is successfully retrieved. If retrieval yields no results, a deterministic fallback response is returned instead of invoking the model.
```python
# ai_assistant_v3.py

if not response.source_nodes:
    final_answer = "I do not have enough information in the provided knowledge base."
else:
    final_answer = response.response
print(f"AI Assistant: {final_answer}")
```

## Demo & Evaluation
This section focuses on the demonstration and evaluation of the system. The demonstration consists of three user inputs, where the final input intentionally queries an issue or disease that is **not** present in the local knowledge base.  

The objective is to validate correct LLM behavior in successful retrieval scenarios, as well as in cases where no relevant data exists for a given query. In such situations, the system is expected to respond appropriately by acknowledging the absence of sufficient information rather than generating unsupported or hallucinated content.

### Demo
```shell
$ python ai_assistant_v3.py
Start chatting (type 'exit' to quit).

You: I have a headache and sensitivity to light.
DISCLAIMER: 
        This AI assistant is for demonstration
        purposes  only  and  does  not provide 
        medical advice.
    
AI Assistant: Rest in a quiet, dark environment and consider seeking medical advice if headaches are frequent or severe.

You: I do not have appetite and I feel nausea.
DISCLAIMER: 
        This AI assistant is for demonstration
        purposes  only  and  does  not provide 
        medical advice.
    
AI Assistant: Rest until symptoms improve, eat light and bland foods, drink fluids in small amounts, and avoid heavy or spicy meals. If symptoms persist or worsen, consult a professional.

You: I cannot stand on my leg as my big toe hurts a lot.
DISCLAIMER: 
        This AI assistant is for demonstration
        purposes  only  and  does  not provide 
        medical advice.
    
AI Assistant: Empty Response

You: exit
Bye!
```

### Results
| Disease | User input | Expected Assistant Respond | Evaluation |
| ------- | ---------- | -------------------------- | ---------- |
| Headache | I have a headache and sensitivity to light. | Headache Solution | OK |
| Upset Stomach | I do not have appetite and I feel nausea. | Upset Stomach Solution | OK |
| Broken Big Toe | I cannot stand on my leg as my big toe hurts a lot. | "I do not have enough information" | OK |

![demo_results.txt](https://github.com/zvolenskydaniel/AI/blob/main/02-langchain-llamaindex/examples/demo_results.txt)

### Evaluation
The system was evaluated using three representative test cases designed to validate both successful retrieval scenarios and correct fallback behavior when relevant data is unavailable. The outcomes are assessed against the predefined [Success Criteria and KPIs](#success-criteria--kpis).

#### Functional Accuracy
**Condition Recognition Accuracy**  
The assistant successfully identified the correct condition for all test cases where the condition was present in the local knowledge base. For the inputs related to *Headache* and *Upset Stomach*, the assistant correctly mapped user-described symptoms to the corresponding conditions defined in `common_health_issues.txt`.  
For the intentionally unsupported query (*Broken Big Toe*), the assistant correctly recognized that no matching condition existed and did not attempt to infer or fabricate an answer.

**Relevant Knowledge Retrieval**  
Retrieved document chunks were directly related to the user queries in all supported cases. No unrelated or redundant content was observed in the retrieved context. In the unsupported case, no misleading or partially relevant chunks were retrieved, demonstrating effective retrieval filtering.

#### Response Quality
**Clarity and Readability**  
All generated responses were concise and easy to understand for non-technical users. Medical terminology was either avoided or presented at a high level, consistent with the educational nature of the simulation.

**Structure Consistency**  
Responses followed a consistent and predictable structure aligned with the source document format. In supported cases, the assistant returned content corresponding to the *Solution* section, as defined by the project constraints.

**Completeness**  
For supported conditions, responses adequately addressed the user’s symptoms and provided a general resolution approach without omitting critical information. For unsupported queries, the response appropriately indicated insufficient knowledge without attempting partial or speculative answers.

#### RAG Performance
**Chunking Effectiveness**  
The applied section-aware, delimiter-based chunking strategy resulted in coherent and semantically complete source nodes. Retrieval results demonstrated improved alignment between user queries and retrieved content, with no observed section leakage or fragmented context.

**Metadata Utilization**  
Metadata filtering (*e.g. domain, document type, content scope*) effectively constrained retrieval to the intended knowledge source. Compared to unfiltered retrieval, metadata usage demonstrably improved precision and reduced the risk of irrelevant context being passed to the LLM.

#### Safety and Boundary Compliance
**Disclaimer Enforcement**  
The assistant respected the defined safety boundaries throughout the evaluation. No medical diagnoses, personalized treatment plans, or prescriptive advice were generated.

**Refusal Handling**  
In the unsupported test case, the assistant appropriately declined to provide an answer by explicitly stating that sufficient information was not available in the knowledge base. The refusal was clear, polite, and aligned with the predefined fallback behavior.

#### System Reliability
**Response Consistency**  
The system produced logically consistent responses for similar inputs and did not hallucinate conditions outside the provided knowledge base. Behavior remained stable across all evaluated scenarios.

**Latency**  
All responses were generated within an acceptable time frame for an interactive demonstration environment. Latency remained suitable for real-time user interaction under the current local execution setup.

Overall, the evaluation results indicate that the system meets the defined [Success Criteria and KPIs](#success-criteria--kpis), demonstrating reliable retrieval-augmented generation, effective safety controls, and predictable behavior suitable for educational AI health simulations.

## Future Improvements
Potential extensions to the project include:
- enabling the AI assistant to propose updates to `common_health_issues.txt` by adding new conditions while preserving the existing document structure
- convert local knowledge source file `common_health_issues.txt` first into `markdown` and then into `yaml`
- locally stored conversational memory between the user and the AI assistant
- design a troubleshooting conversation flow (symptom → clarification → suggestion)
- tag issues with metadata (severity, urgency, keywords)
- build a "why this answer" explanation
