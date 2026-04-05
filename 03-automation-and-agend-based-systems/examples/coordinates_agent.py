#
# 2026.04 AI: Learning Path
# zvolensky.daniel@gmail.com
#

# ---- Import libraries ----
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Load OPEN_AI_KEY
load_dotenv()

# Get coordinates: Coordinates API
# @tool decorator automatically does the input typing + metadata LangChain needs.
@tool
def geocode_city(city: str) -> dict:
    """
    Convert city name to latitude and longitude.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": city, 
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "langchain-agent-demo"
    }

    response = requests.get(url, params=params, headers=headers, timeout=10)
    data = response.json()

    if not data:
        return {"error": "City not found"}

    return {
        "latitude": float(data[0]["lat"]),
        "longitude": float(data[0]["lon"]),
    }

# Define LLM
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0
)

# Create the Agent
agent = create_agent(
    model = llm,
    tools = [geocode_city]
)

# Run it
ai_message = agent.invoke({
    "messages": [
        HumanMessage(content="What's the coordinates for the city Bratislava?")
    ]
})

print(ai_message["messages"][-1].content)
