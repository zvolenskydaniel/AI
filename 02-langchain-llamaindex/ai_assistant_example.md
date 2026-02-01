# Example: AI Assistant Using Python Libraries LlamaIndex and LangChain

## Table of content
- [Objective](#objective)
- [Legal & Safety Disclaimer]()
- [Demonstration & Evaluation]()
- [System Reliability]()
- [Development Requirements]()
- [Future Improvements]()

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

## Future Improvements
Potential extensions to the project include:
- convert local knowledge source file `common_health_issues.txt` first into `markdown` and then into `yaml`
- locally stored conversational memory between the user and the AI assistant
- enabling the AI assistant to propose updates to `common_health_issues.txt` by adding new conditions while preserving the existing document structure

Next points:
[X] convert this into YAML / JSON / Markdown for LlamaIndex
[ ] add intent-style phrasing (how users might ask questions)
[ ] design a troubleshooting conversation flow (symptom → clarification → suggestion)
[ ] tag issues with metadata (severity, urgency, keywords)

[ ] inspect retrieved nodes during a query
[ ] add metadata filters to the retriever
[ ] enforce “answer must come from this file”
[ ] add a tiny evaluation harness for chunking strategies

[ ] build a side-by-side chunking comparison
[ ] add automatic hallucination detection
