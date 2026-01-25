#
# 2026.01 AI: Learning Path
# zvolensky.daniel@gmail.com
#

import logging
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# Set logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)

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
