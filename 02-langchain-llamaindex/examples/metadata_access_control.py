#
# 2026.01 AI: Learning Path
# zvolensky.daniel@gmail.com
#

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
