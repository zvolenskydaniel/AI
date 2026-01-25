#
# 2026.01 AI: Learning Path
# zvolensky.daniel@gmail.com
#

import os
from llama_index.core import SimpleDirectoryReader

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

# Inspect loaded documents
doc = documents[0]

print("Document text preview:")
print(doc.text[:200])

print("\nDocument metadata:")
print(doc.metadata)
