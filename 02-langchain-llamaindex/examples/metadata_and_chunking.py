#
# 2026.01 AI: Learning Path
# zvolensky.daniel@gmail.com
#

import logging
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.llms.openai import OpenAI

# Set logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)

# Load .env
load_dotenv()

# LLM model definition
Settings.llm = OpenAI(
    model = "gpt-4o-mini",
    temperature = 0.0
)

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

# Define chunking strategy (sentence-based)
splitter = SentenceSplitter(
    chunk_size = 300,
    chunk_overlap = 50,
)

# Build vector index + apply chunking strategy
index = VectorStoreIndex.from_documents(
    documents,
    transformations = [splitter]
)

# Inspect chunks and metadata
nodes = splitter.get_nodes_from_documents(documents)
for node in nodes:
    print(f"Chunk text:\n{node.text}\n")
    print(f"Metadata: {node.metadata}")

# Define metadata filters explicitly
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

# Metadata filtering during quering
query_engine = index.as_query_engine(filters = filters)

# Query -> Response
response = query_engine.query(
    "How many days per week can employees work remotely?"
)

# Print output
print(f"\nResponse: {response}\n")
