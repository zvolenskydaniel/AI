#
# 2026.01 AI: Learning Path
# zvolensky.daniel@gmail.com
#

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

# Send request to the LLM
response = client.responses.create(
    model = "gpt-4o-mini",
    instructions = "You are a helpful assistant.",
    input = "What's capital of Slovakia?"
)

# Print model output
print(response.output_text)
