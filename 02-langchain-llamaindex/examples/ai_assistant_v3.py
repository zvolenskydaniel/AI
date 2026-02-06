#
# 2026.02 AI: Learning Path
# zvolensky.daniel@gmail.com
#

import logging
import os
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI

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

# LLM model definition
Settings.llm = OpenAI(
    model = "gpt-4o-mini",
    temperature = 0.0
)

# Get the directory where the script itself is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Combine that directory with the filename
file_path = os.path.join(script_dir, "data", "common_health_issues.txt")

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
    similarity_top_k = 2,
    response_mode = "compact",
    text_qa_template = PromptTemplate(text_qa_template_str),
    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff = 0.75)
    ]
)

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
    if not response.source_nodes:
        final_answer = "I do not have enough information in the provided knowledge base."
    else:
        final_answer = response.response
    print(f"AI Assistant: {final_answer}")

    logger.info(f"User: {user_query}")
    logger.info(f"AI Assistant: {final_answer}")

    # ------------------------------------------------------------------
    # Inspect source nodes
    logger.info(f"=== Source Nodes ===")
    for i, node_with_score in enumerate(response.source_nodes, start = 1):
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
