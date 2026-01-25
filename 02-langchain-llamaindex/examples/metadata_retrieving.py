#
# 2026.01 AI: Learning Path
# zvolensky.daniel@gmail.com
#

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
