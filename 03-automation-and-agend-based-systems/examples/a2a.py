#
# 2026.04 AI: Learning Path
# zvolensky.daniel@gmail.com
#

# ---- Import libraries ----
import json
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

# Load OPEN_AI_KEY
load_dotenv()

# Define LLM
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0
)

#---------------------------------------------------------------
# Coordinates Agent

# Extract coordinates
def extract_coords(agent_response):
    for msg in agent_response["messages"]:
        if isinstance(msg, ToolMessage):
            content = msg.content
            # ToolMessage.content is a JSON string
            if isinstance(content, str):
                return json.loads(content)
            return content
    raise ValueError("No tool output found for coordinates.")

# Get coordinates: Coordinates API
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

# Create the Agent
geo_agent = create_agent(
    model = llm,
    tools = [geocode_city]
)

#---------------------------------------------------------------
# Weather Agent

# Get weather: Weather API
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

# Create the Agent
weather_agent = create_agent(
    model = llm,
    tools = [get_weather]
)

#---------------------------------------------------------------
# Simple Agent-to-Agent Logic (manual orchestration)

def weather_by_city(city: str) -> str:
    """
    Call each agent and return current weather for specific city.
    """
    # print(f"city: {city}")
    coords_raw = geo_agent.invoke({
        "messages": [
            HumanMessage(
                content = f"Get coordinates for the city {city}."
            )
        ]
    })
    # extract latitude & longitude received from coordinates_agent
    coords = extract_coords(coords_raw)
    # print(f"coords: {coords}")
    latitude = coords["latitude"]
    longitude = coords["longitude"]
    weather_raw = weather_agent.invoke({
        "messages": [
            HumanMessage(
                content=f"Provide the current weather for {city} based on these coordinates: "
                        f"latitude {latitude} and longitude {longitude}. "
                        f"Output ONLY the weather in this format: "
                        f"'The current weather in [City] is [Temp]°C with a wind speed of [Speed] km/h.' "
            )
        ]
    })
    weather = weather_raw["messages"][-1].content
    return weather

print(weather_by_city(city = "Bratislava"))
