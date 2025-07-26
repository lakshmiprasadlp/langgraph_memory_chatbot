from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Load memory checkpointer
memory = MemorySaver()

# Define state
class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

# Set up LLM
llm = ChatOpenAI(model="gpt-4o")  # You can switch to "gpt-3.5-turbo" if needed

# Define chatbot logic
def chatbot(state: BasicChatState): 
    return {
        "messages": [llm.invoke(state["messages"])]
    }

# Build the LangGraph app
def build_graph():
    graph = StateGraph(BasicChatState)
    graph.add_node("chatbot", chatbot)
    graph.add_edge("chatbot", END)
    graph.set_entry_point("chatbot")
    return graph.compile(checkpointer=memory)
