
# agent_langchain.py
# Study Buddy - AI Agent using LangChain

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from tools import (
    SYSTEM_PROMPT,
    search_notes,
    calculate,
    today,
    add_task
)


# Load environment variables
load_dotenv()

# Check API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")


# Create the language model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=api_key
)


# Create the AI agent
agent = create_agent(
    model=llm,
    tools=[
        search_notes,
        calculate,
        today,
        add_task
    ],
    system_prompt=SYSTEM_PROMPT
)


# Function to ask questions
def ask_agent(question):

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    messages = result["messages"]

    # Find the tools used by the agent
    steps = [
        call["name"]
        for message in messages
        for call in getattr(message, "tool_calls", [])
    ]

    # Get final answer
    answer = messages[-1].content

    return answer, steps


# Start the agent
if __name__ == "__main__":

    print("\n===== STUDY BUDDY AI AGENT =====")
    print("Type 'quit' to exit.")

    while True:

        question = input("\nYou: ")

        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        try:
            answer, steps = ask_agent(question)

            print("\nTools used:", steps or "none")
            print("\nAgent:", answer)

        except Exception as e:
            print("\nError:", e)
