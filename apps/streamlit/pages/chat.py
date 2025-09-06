import streamlit as st
import sys
import os

# Make repo root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.llm.Retrivals.text_embedding_retrival import chroma_retrieve

st.set_page_config(page_title="Ask Questions", page_icon="💬", layout="wide")
st.title("Chat with your Data 📃")

# --- Reset chat when the active file changes -----------------------------------
# Uses 'last_file_id' that your uploader sets in session_state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("conversation_file_key", None)

current_file_key = st.session_state.get("last_file_id")  # set in process_uploaded_file(...)
if st.session_state["conversation_file_key"] != current_file_key:
    # New (or no) file => start a fresh conversation
    st.session_state["messages"].clear()
    st.session_state["conversation_file_key"] = current_file_key

# ------------------------------------------------------------------------------

# Show context (filename if available)
if "original_filename" in st.session_state and st.session_state["original_filename"]:
    st.info(f"Chatting over 📂 **{st.session_state['original_filename']}**")

# (Optional) Manual reset button
# if st.button("🧹 New chat", use_container_width=False):
#     st.session_state["messages"].clear()
#     st.rerun()

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

    # Guard: what source are we using?
    selected = st.session_state.get("selected_source")
    is_file_source = selected in ("PDF", "DOCX", "CSV")

    response = "I don't have a file context yet. Please upload a PDF/DOCX/CSV."
    if is_file_source:
        file_id = st.session_state.get("last_file_id")  # set by uploader flow
        if file_id:
            response = chroma_retrieve(
                prompt,
                top_k=20,           # dense & bm25 each fetch ~20, then fusion + rerank->8
                use_mmr=True,
                mmr_lambda=0.5,
                file_id=file_id,    # important for hybrid BM25 to stay on this file
                rerank_top_n=8,
            )
        else:
            response = "I see a file source selected, but no file is loaded yet."

    # Assistant bubble
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
