import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.llm.Retrivals.text_embedding_retrival import chroma_retrieve

st.set_page_config(page_title="Ask Questions", page_icon="💬", layout="wide")
st.title("Chat with your Data 📃")

# Init history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show context
if "original_filename" in st.session_state:
    st.info(f"Chatting over 📂 **{st.session_state['original_filename']}**")

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask a question about your data..."):
    # User bubble
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # guard: what source are we using?
    selected = st.session_state.get("selected_source")
    # you likely store it as a string, not a list
    is_file_source = selected in ("PDF", "DOCX", "CSV")

    response = "I don't have a file context yet. Please upload a PDF/DOCX/CSV."
    if is_file_source:
        file_id = st.session_state.get("last_file_id")  # you set this in your uploader flow
        response = chroma_retrieve(
            prompt,
            top_k=20,             # dense & bm25 each fetch ~20, then fusion + rerank->8
            use_mmr=True,
            mmr_lambda=0.5,
            file_id=file_id,      # important for hybrid BM25 to stay on this file
            rerank_top_n=8,
        )

    # print(response)
    
    # Assistant bubble
    st.session_state.messages.append({"role": "assistant", "content": response})


    with st.chat_message("assistant"):
        st.markdown(response)
