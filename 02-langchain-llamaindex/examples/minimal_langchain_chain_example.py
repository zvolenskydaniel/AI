#
# 2026.01 AI: Learning Path
# zvolensky.daniel@gmail.com
#

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage

# Load .env
load_dotenv()

# Initialize model
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.2
)

# Input/User data
COUNTRY = "Slovakia"
PROMPT = f"What's capital of {COUNTRY}?"

# Define System and Human messages
messages = [
    SystemMessage(content = "You are a helpful assistant."),
    HumanMessage(content = PROMPT),
]

# Invoke returns an AIMessage
ai_message = llm.invoke(messages)

# Print model output
print(ai_message.content)
