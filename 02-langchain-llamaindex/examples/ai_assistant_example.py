#
# 2026.02 AI: Learning Path
# zvolensky.daniel@gmail.com
#

import logging
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
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

# Define chunking strategy (Section-Aware Chunking)
splitter = SentenceSplitter(
    chunk_size = 600,
    chunk_overlap = 100,
)

# Build vector index + apply chunking strategy
index = VectorStoreIndex.from_documents(
    documents,
    transformations = [splitter]
)

# TODO: choose option for isnpection, add them to logs for cleaner output?

#-----------------------------------------------------------
# Option 1 (Best): Inspect Retrieved Nodes via the Retriever
#-----------------------------------------------------------
# Create retriever from existing index
retriever = index.as_retriever(similarity_top_k = 3)

# Define test query
query = "I have a headache and sensitivity to light."

# Retrieve nodes (NO re-chunking)
retrieved_nodes = retriever.retrieve(query)

# Inspect retrieved nodes
for i, node_with_score in enumerate(retrieved_nodes, start = 1):
    node = node_with_score.node
    score = node_with_score.score

    print(f"\n--- Retrieved Node {i} ---")
    print(f"Score: {score:.4f}")
    print(f"Metadata: {node.metadata}")
    print(f"Text:\n{node.text}")

#-----------------------------------------------------------
# Option 2: Inspect Nodes Used by the Query Engine
#-----------------------------------------------------------
query_engine = index.as_query_engine(similarity_top_k = 3)

response = query_engine.query("I have a headache and sensitivity to light.")

# Final LLM answer
print("\n=== Final Answer ===")
print(response.response)

# Inspect source nodes
print("\n=== Source Nodes ===")
for i, node_with_score in enumerate(response.source_nodes, start=1):
    node = node_with_score.node
    score = node_with_score.score

    print(f"\n--- Source Node {i} ---")
    print(f"Score: {score:.4f}")
    print(f"Metadata: {node.metadata}")
    print(f"Text:\n{node.text}")

"""
- debugging hallucinations
- verifying “answer came from source”
- demonstrating RAG transparency to others


# Difference Between Retriever vs Query Engine

| Aspect                       | Retriever | Query Engine |
| ---------------------------- | --------- | ------------ |
| Chunk inspection             | ✅        | ✅            |
| Similarity scores            | ✅        | ✅            |
| Shows LLM output             | ❌        | ✅            |
| Shows what LLM actually used | ❌        | ✅            |
| Best for tuning retrieval    | ✅        | ⚠️           |

"""

#-----------------------------------------------------------
# Option 3 (Advanced): Metadata-Filtered Debugging
#-----------------------------------------------------------

filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="domain",
            value="health_simulation"
        )
    ]
)

retriever = index.as_retriever(
    similarity_top_k=3,
    filters=filters
)

nodes = retriever.retrieve("burn on my hand")
