#
# 2026.02 AI: Learning Path
# zvolensky.daniel@gmail.com
#

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# Load .env
load_dotenv()

# Initialize model
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.2
)

# Create the Agent
agent = create_agent(
    model = llm,
    tools = [],
    system_prompt = "You are a helpful assistant.",
    middleware = [
        SummarizationMiddleware(
            model = llm,
            trigger = ("tokens", 4000),
            keep = ("messages", 20)
        )
    ],
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
