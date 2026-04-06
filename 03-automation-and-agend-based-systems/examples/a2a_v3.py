#
# 2026.04 AI: Learning Path
# zvolensky.daniel@gmail.com
#

import json
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load OPEN_AI_KEY
load_dotenv()

# Define LLM
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0
)

# --- Define Schema/Formating ---

class GeoResponse(BaseModel):
    latitude: float = Field(description="Latitude of the city")
    longitude: float= Field(description="Longitude of the city")

class WeatherResponse(BaseModel):
    temperature: float = Field(description="Current temperature in Celsius")
    wind_speed: float = Field(description="Current wind speed in km/h")

# --- Tool Definition ---
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

# Get weather: Weather API
@tool
def weather_city(latitude: float, longitude: float) -> str:
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

def weather_chain(city: str):
    """Unified Orchestration Logic"""
    geo_llm = llm.bind_tools([geocode_city]).with_structured_output(GeoResponse)
    weather_llm = llm.bind_tools([weather_city]).with_structured_output(WeatherResponse)

    coords = geo_llm.invoke(f"Find coordinates for {city}")
    weather_result = weather_llm.invoke(
        f"Get weather for lat: {coords.latitude}, lon: {coords.longitude}"
    )

    return {
        "location": city,
        "coordinates": coords.model_dump(),
        "weather": weather_result.model_dump()
    }

# --- Execution ---
result = weather_chain("Bratislava")
print(json.dumps(result, indent=2))
