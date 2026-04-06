#
# 2026.04 AI: Learning Path
# zvolensky.daniel@gmail.com
#

import requests
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load OPEN_AI_KEY
load_dotenv()

# --- Define LLM ---
llm = LLM(
    model = "gpt-4o-mini",
    temperature = 0
)

# --- Define Tools Classes ---
class GeocodeInput(BaseModel):
    city: str = Field(description="The name of the city to search for (e.g., 'Bratislava').")

class GeocodeTool(BaseTool):
    name: str = "geocode_city"
    description: str = "Convert a city name into latitude and longitude coordinates."
    args_schema: type[BaseModel] = GeocodeInput

    # Get coordinates: Coordinates API
    def _run(self, city: str) -> str:
        """Convert city name to latitude and longitude."""
        print(f"DEBUG: Agent is searching for: {city}")

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": city, 
            "format": "json",
            "limit": 1
        }
        headers = {
            "User-Agent": "langchain-agent-demo"
        }
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()

            if not data:
                return f"Error: City '{city}' not found. Please try a different name."

            return f"Latitude: {float(data[0]['lat'])}, Longitude: {float(data[0]['lon'])}"

        except Exception as e:
            return f"Error connecting to Geocoding service: {str(e)}"

class WeatherInput(BaseModel):
    latitude: float = Field(description="The latitude coordinate.")
    longitude: float = Field(description="The longitude coordinate.")

class WeatherTool(BaseTool):
    name: str = "weather_city"
    description: str = "Get current weather for given latitude and longitude coordinates."
    args_schema: type[BaseModel] = WeatherInput

    # Get weather: Weather API
    def _run(self, latitude: float, longitude: float) -> str:
        """Get current weather for given coordinates."""
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}&current_weather=true"
        )

        try:
            response = requests.get(url, timeout=10)
            data = response.json()

            # Another safety check
            if "current_weather" not in data:
                return "Error: Weather data currently unavailable for these coordinates."

            weather = data.get("current_weather", {})
            return f"Temperature: {weather.get('temperature')}°C, Wind: {weather.get('windspeed')} km/h"

        except Exception as e:
            return f"Error connecting to Weather service: {str(e)}"

geocode_tool = GeocodeTool()
weather_tool = WeatherTool()

# --- Define the Agents ---
geo_researcher = Agent(
    role='Geographic Researcher',
    goal='Find accurate coordinates for {city}',
    backstory='Expert in geocoding and global coordinates.',
    tools=[geocode_tool],
    llm=llm,
    verbose=True
)

weather_analyst = Agent(
    role='Weather Analyst',
    goal='Provide weather data for coordinates provided by the researcher',
    backstory='Meteorological data specialist.',
    tools=[weather_tool],
    llm=llm,
    verbose=True
)

# --- Define the Tasks ---
# CrewAI tasks can pass context to each other automatically
coord_task = Task(
    description='Find the latitude and longitude for the city: {city}',
    expected_output='A dictionary with latitude and longitude.',
    agent=geo_researcher
)

weather_task = Task(
    description='Look at the coordinates from the previous task and provide the current weather.',
    expected_output='A concise weather report including temperature and wind speed.',
    agent=weather_analyst,
    context=[coord_task] # This tells CrewAI to feed the output of task 1 into task 2
)

# --- Assemble the Crew ---
weather_crew = Crew(
    agents=[geo_researcher, weather_analyst],
    tasks=[coord_task, weather_task],
    process=Process.sequential # Task 1 must finish before Task 2 starts
)

# --- Execution ---
result = weather_crew.kickoff(inputs={'city': 'Bratislava'})

print("\n--- FINAL CREW REPORT ---")
print(result)
