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

# Load documents from disk
documents = SimpleDirectoryReader(file_path).load_data()

# Print output
print(f"Loaded {len(documents)} document(s)")

# Inspecting loaded documents
doc = documents[0]

print("Text preview:")
print(doc.text[:300])

print("\nMetadata:")
print(doc.metadata)
