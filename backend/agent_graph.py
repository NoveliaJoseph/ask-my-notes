
# agent_graph.py
# Study Buddy - AI Agent using LangGraph

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from tools import (
    SYSTEM_PROMPT,
    search_notes,
    calculate,
    today,
    add_task
)


# STEP 1: Load environment variables
load_dotenv()


# STEP 2: Available tools
tools = [
    search_notes,
    calculate,
    today,
    add_task
]


# STEP 3: Create AI model and connect tools
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
).bind_tools(tools)


# STEP 4: THINK NODE
def think(state: MessagesState):

    # AI reads the conversation and decides what to do
    reply = llm.invoke(
        [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
    )

    return {"messages": [reply]}


# STEP 5: CREATE THE GRAPH
graph = StateGraph(MessagesState)


# STEP 6: Add nodes
graph.add_node("think", think)
graph.add_node("tools", ToolNode(tools))


# STEP 7: Connect nodes
graph.add_edge(START, "think")

graph.add_conditional_edges(
    "think",
    tools_condition
)

graph.add_edge("tools", "think")


# STEP 8: Compile graph with memory
study_graph = graph.compile(
    checkpointer=InMemorySaver()
)


# STEP 9: START THE AGENT
if __name__ == "__main__":

    # One memory for this conversation
    chat = {
        "configurable": {
            "thread_id": "student-1"
        }
    }

    print("\n===== STUDY BUDDY LANGGRAPH AGENT =====")
    print("Type 'quit' to exit.")

    while True:

        question = input("\nYou: ")

        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # Send user message to the graph
        user_msg = {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }

        # Run graph and show each step
        try:

            for event in study_graph.stream(
                user_msg,
                chat,
                stream_mode="updates"
            ):

                for node, update in event.items():

                    last = update["messages"][-1]

                    if node == "tools":
                        print(
                            f"👀 [tools] {last.content[:80]}"
                        )

                    elif last.tool_calls:
                        print(
                            "🛠 [think] wants:",
                            [call["name"] for call in last.tool_calls]
                        )

                    else:
                        print("Agent:", last.content)

        except Exception as e:
            print("Error:", e)
