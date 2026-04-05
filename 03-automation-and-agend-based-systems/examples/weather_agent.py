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

# Get weather: Weather API
# @tool decorator automatically does the input typing + metadata LangChain needs.
@tool
def get_weather(latitude: float, longitude: float) -> str:
    """
    Get current weather for given coordinates.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}&current_weather=true"
    )
    response = requests.get(url, timeout=10)
    data = response.json()

    weather = data.get("current_weather", {})
    return (
        f"Temperature: {weather.get('temperature')}°C,\n"
        f"Wind: {weather.get('windspeed')} km/h"
    )

# Define LLM
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0
)

# Create the Agent
agent = create_agent(
    model = llm,
    tools = [get_weather]
)

# Run it
ai_message = agent.invoke({
    "messages": [
        HumanMessage(content="What's the weather at latitude 48.1559 longitude 17.1314?")
    ]
})

print(ai_message["messages"][-1].content)
