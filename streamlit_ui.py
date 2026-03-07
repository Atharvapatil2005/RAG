import streamlit as st
import requests

st.title("RAG Chat")

query=st.text_input("Ask something: ")
#the ui now does not expect json 
if query:
    response=requests.post(
        "http://localhost:8000/chat",
        json={"question":query},
        stream = True
    )
    placeholder=st.empty()
    full_text=""

    try:
        for chunk in response.iter_content(chunk_size=1):
            if chunk:
                token=chunk.decode("utf-8")
                full_text += token
                placeholder.markdown(full_text)
    except requests.exceptions.ChunkedEncodingError:
        pass
    
    