#
# 2026.01 AI: Learning Path
# zvolensky.daniel@gmail.com
#

import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Get the directory where the script itself is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Combine that directory with the filename
file_path = os.path.join(script_dir, "history.json")

# Load .env
load_dotenv()

# Initialize model
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.2
)

# Create empty list to store conversation history
history = []  # list of (role, message) tuples

# Get previous conversations: read the file history.json
with open(file_path, "r") as file:
    data = json.load(file)
    history = [
        HumanMessage(content = item["content"]) if item["role"] == "human"
        else AIMessage(content = item["content"])
        for item in data
    ]

# Build prompt template that allows inserting history
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content = "You are a helpful assistant who respond in max. 4 words."),
    MessagesPlaceholder(variable_name = "history"),
    HumanMessage(content = "{user_input}")
])

# Input/User data
user_inputs = ["What is my name?", "What is my favourite activity?", "quit"]

for user_text in user_inputs:
    # Print user's input
    print("User:", user_text)

    # Capture the conversation into history.json
    if user_text.lower() in ("exit", "quit", "q"):
        serializable = [{"role": msg.type, "content": msg.content} for msg in history]
        with open(file_path, "w") as file:
            json.dump(serializable, file, indent=4)
        print("Conversation saved. Bye!")
        break

    # Append user message to history
    history.append(HumanMessage(content = user_text))

    # Format prompt with messages
    prompt_messages = prompt.format_messages(
        history = history,
        user_input = user_text
    )

    # Invoke LLM
    response = llm.invoke(prompt_messages)

    # Get response content
    content = response.content if hasattr(response, "content") else str(response)
    print("Assistant:", content)

    # Append assistant message to history
    history.append(AIMessage(content = response.content))
