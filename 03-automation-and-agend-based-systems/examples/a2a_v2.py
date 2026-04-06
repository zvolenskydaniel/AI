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
from pydantic import BaseModel, Field

# Load OPEN_AI_KEY
load_dotenv()

# Define LLM
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0
)

def tool_output(response):
    """Extract the tool output"""
    raw_content = ""
    for msg in response["messages"]:
        if isinstance(msg, ToolMessage):
            raw_content = msg.content
    return raw_content

#---------------------------------------------------------------
# Coordinates Agent

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
# Define Schema: formating of the expected result/repsond

class GeoResponse(BaseModel):
    latitude: float = Field(description="Latitude of the city")
    longitude: float = Field(description="Longitude of the city")

class WeatherResponse(BaseModel):
    city: str = Field(description="The name of the city")
    temperature: float = Field(description="Current temperature in Celsius")
    wind_speed: float = Field(description="Current wind speed in km/h")
    summary: str = Field(description="A one-sentence summary of the weather")

#---------------------------------------------------------------
# Simple Agent-to-Agent Logic (manual orchestration)

def weather_by_city(city: str):
    """
    Call each agent and return current weather for specific city.
    """
    coords_raw = geo_agent.invoke({
        "messages": [
            HumanMessage(
                content = f"Use your tools to find coordinates for {city}"
            )
        ]
    })
    raw_content = tool_output(response = coords_raw)

     # Structured output
    structured_llm_geo = llm.with_structured_output(GeoResponse)

    # Pass raw API data to the LLM
    prompt_geo = f"Extract coordinates from this raw data: {raw_content}"
    coords = structured_llm_geo.invoke(prompt_geo)
    
    # Get Weather Data
    weather_raw = weather_agent.invoke({
        "messages": [
            HumanMessage(
                content = f"Get weather for latitude {coords.latitude} and longitude {coords.longitude}."
            )
        ]
    })
    weather_data = weather_raw["messages"][-1].content
    
    # Wrap the LLM with the structured output requirement
    structured_llm_wea = llm.with_structured_output(WeatherResponse)

    # Use the LLM as a "Parser" to fill the Pydantic object
    prompt_wea = f"Format the weather data for {city}. Data: {weather_data}"
    
    # Return WeatherResponse
    final_output = structured_llm_wea.invoke(prompt_wea)
    
    return final_output

# --- Execution ---
result = weather_by_city("Bratislava")

# Now you can access it like an object:
print(f"City: {result.city}")
print(f"Temp: {result.temperature}")

# Or convert to a clean JSON string:
print(result.model_dump_json(indent = 2))
