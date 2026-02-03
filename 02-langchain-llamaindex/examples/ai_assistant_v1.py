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
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate

# --- Configure the logging module ---
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Set the log message format
    filename='./examples/logger/logger.log',  # Specify the log file
)
# --- Create a logger instance ---
logger = logging.getLogger(__name__)

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
    text_qa_template = PromptTemplate(text_qa_template_str),
    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff = 0.75)
    ]
)

# Query -> Response
user_query = "I have a headache and sensitivity to light."
response = query_engine.query(user_query)

# Print model output
print("\n=== Final Answer ===")
print(response.response)

logger.info(f"User: {user_query}")
logger.info(f"AI Assistant: {response.response}")

# ------------------------------------------------------------------
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
