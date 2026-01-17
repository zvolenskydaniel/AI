#
# 2026.01 AI: Learning Path
# zvolensky.daniel@gmail.com
#

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

prompt = """
Return the capital of Slovakia in JSON format.

Expected format:
{
  "country": "<string>",
  "capital": "<string>"
}
"""

# Send request to the LLM
response = client.responses.create(
    model = "gpt-4o-mini",
    instructions="You are a precise assistant that outputs valid JSON only.",
    input = prompt,
    temperature = 0.0
)

# Print model output
raw_output = response.output_text
print("Raw model output:")
print(raw_output)

# Remove markdown code fences if present
if raw_output.startswith("```"):
    raw_output = raw_output.split("```")[1].strip()
    raw_output = raw_output.split("json")[1].strip()

data = json.loads(raw_output)
print("\nParsed JSON:")
print(data)
