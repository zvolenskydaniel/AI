#
# 2026.01 AI: Learning Path
# zvolensky.daniel@gmail.com
#

import os
import openai
import json
import time
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

def retry_llm_call(fn, retries=3, backoff=2):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except RuntimeError as e:
            if attempt == retries:
                raise
            time.sleep(backoff ** attempt)

def get_capital(country: str) -> dict:
    prompt = f"""
    Return the capital of {country} in JSON format.

    Expected format:
    {{
      "country": "<string>",
      "capital": "<string>"
    }}
    """

    try:
        # Send request to the LLM
        response = client.responses.create(
            model = "gpt-4o-mini",
            instructions="Return valid JSON only. Do not include explanations.",
            input = prompt,
            temperature = 0.0
        )

        # Get LLM output
        data = json.loads(response.output_text)
        
        # Validate output
        validate(
            instance = data,
            schema = schema
        )
        return data

    except json.JSONDecodeError:
        print("Invalid JSON returned by model.")

    except ValidationError as e:
        print(f"JSON schema validation failed: {e}.")

    except openai.APIError as e:
      #Handle API error here, e.g. retry or log
      print(f"OpenAI API returned an API Error: {e}")
      pass
    except openai.APIConnectionError as e:
      #Handle connection error here
      print(f"Failed to connect to OpenAI API: {e}")
      pass
    except openai.RateLimitError as e:
      #Handle rate limit error (we recommend using exponential backoff)
      print(f"OpenAI API request exceeded rate limit: {e}")
      pass

result = retry_llm_call(lambda: get_capital(country = "Slovakia"))
print(result)
