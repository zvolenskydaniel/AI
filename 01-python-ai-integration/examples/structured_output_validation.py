#
# 2026.01 AI: Learning Path
# zvolensky.daniel@gmail.com
#

import os
import json
from jsonschema import validate, ValidationError
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

# Define expected schema
schema = {
    "type": "object",
    "properties": {
        "country": {"type": "string"},
        "capital": {"type": "string"}
    },
    "required": ["country", "capital"],
    "additionalProperties": False
}

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
    instructions="Return valid JSON only. Do not include explanations.",
    input = prompt,
    temperature = 0.0
)

# Print model output
raw_output = response.output_text
try:
    data = json.loads(raw_output)
    validate(
        instance = data,
        schema = schema
    )
    print("Validated output:")
    print(data)

except json.JSONDecodeError:
    print("Invalid JSON returned by model")

except ValidationError as e:
    print("JSON schema validation failed:")
    print(e)
