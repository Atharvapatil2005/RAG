import streamlit as st
import requests

st.title("RAG Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Ask something:")

if user_input:
    response = requests.post(
        "http://localhost:8000/chat",
        json={"question": user_input}
    )

    answer = response.json()["answer"]

    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("LLM", answer))

for role, msg in st.session_state.messages:
    st.write(f"**{role}:** {msg}")