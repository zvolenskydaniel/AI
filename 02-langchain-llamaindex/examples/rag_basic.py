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
