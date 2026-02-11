#
# 2026.02 AI: Learning Path
# zvolensky.daniel@gmail.com
#

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from typing import Any

# Load OPEN_AI_KEY
load_dotenv()

# Define LLM
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0
)

# --- TRIMMING LOGIC ---
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    Trim conversation history to keep the context window bounded.
    - Keeps only the last MAX_MESSAGES messages.
    - Returns None if no trimming is required.
    """
    MAX_MESSAGES = 4

    messages = state["messages"]

    # Nothing to trim
    if len(messages) <= MAX_MESSAGES:
        return None  # No changes needed

    recent_messages = messages[-MAX_MESSAGES:]
    new_messages = recent_messages

    return {
        "messages": [
            RemoveMessage(id = REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

# Create the Agent with the trimming logic as a state_modifier
agent = create_agent(
    model = llm,
    tools = [],
    system_prompt = "You are a helpful assistant.",
    middleware = [trim_messages],
    checkpointer = InMemorySaver()
)

while True:
    # Get user input
    user_input = input("You: ").strip()
    if user_input.lower() in ("exit", "quit", "q"):
        print("Bye!")
        break

    # Invoke LLM
    ai_message = agent.invoke(
        {
            "messages": [
                HumanMessage(content = user_input)
            ]
        },
        {
            "configurable": {
                "thread_id": "1"
            }
        }
    )

    # Get response content
    content = ai_message["messages"][-1].content
    print("Assistant:", content)
