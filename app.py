import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from graph_config import build_graph

# Load environment variables
load_dotenv()

# Build LangGraph app
app = build_graph()

# Set up Streamlit page
st.set_page_config(page_title="🧠 Memory ChatBot", page_icon="🤖", layout="wide")
st.title("🧠 LangGraph Memory ChatBot")

# Setup thread_id config (can make dynamic later)
config = {
    "configurable": {
        "thread_id": 1
    }
}

# Initialize chat history if not already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box (chat-like)
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Save user message
    st.session_state.chat_history.append(("user", user_input))

    # Get response from LangGraph app
    result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    bot_reply = result["messages"][-1].content

    # Save bot response
    st.session_state.chat_history.append(("bot", bot_reply))

# Render chat messages
for role, message in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(message)
