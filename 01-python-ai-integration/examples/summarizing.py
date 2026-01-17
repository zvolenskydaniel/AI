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

text = """
Slovakia, officially the Slovak Republic, is a landlocked country in Central Europe. It is bordered by Poland to the north, Ukraine to the east, Hungary to the south, Austria to the west, and the Czech Republic to the northwest. Slovakia's mostly mountainous territory spans about 49,000 km2 (19,000 sq mi), hosting a population exceeding 5.4 million. The capital and largest city is Bratislava, while the second largest city is Košice.
"""
prompt = f"Summarize the following text in one sentence:\n{text}"

# Send request to the LLM
response = client.responses.create(
    model = "gpt-4o-mini",
    instructions = "You are a helpful assistant.",
    input = prompt,
    temperature = 0.2
)

# Print model output
print(response.output_text)
