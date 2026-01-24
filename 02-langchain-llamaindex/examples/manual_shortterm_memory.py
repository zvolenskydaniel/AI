#
# 2026.01 AI: Learning Path
# zvolensky.daniel@gmail.com
#

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load .env
load_dotenv()

# Initialize model
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.2
)

# Create empty list to store conversation history
history = []  # list of (role, message) tuples

# Build prompt template that allows inserting history
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content = "You are a helpful assistant who respond in max. 4 words."),
    MessagesPlaceholder(variable_name = "history"),
    HumanMessage(content = "{user_input}")
])

# Input/User data
user_inputs = ["My name is Daniel and I love biking.", "What is my favourite activity?"]

for user_text in user_inputs:
    # print user's input
    print("User:", user_text)

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
