#
# 2026.01 AI: Learning Path
# zvolensky.daniel@gmail.com
#

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent

# Load .env
load_dotenv()

@tool
def get_country_population(country: str) -> str:
    """
    # Return country population
    """
    data = {
        "Slovakia": "5.4 million",
        "Germany": "83 million"
    }
    return data.get(country, "Unknown")

# Initialize model
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.2
)

# Input/User data
COUNTRY = "Slovakia"
PROMPT = f"What's country population of {COUNTRY}?"

# Create the Agent
agent = create_agent(
    model = llm,
    tools = [get_country_population]
)

# Run it
ai_message = agent.invoke({
    "messages": [
        SystemMessage(content = "You are a helpful assistant."),
        HumanMessage(content = PROMPT)
    ]
})

# Print model output
print(ai_message["messages"][-1].content)
