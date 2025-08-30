import streamlit as st

st.set_page_config(page_title="Ask Questions", page_icon="❓", layout="wide")

st.title("💬 AskMyDB - Chat with your Data")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show uploaded file context
if "original_filename" in st.session_state:
    st.info(f"Chatting over 📂 **{st.session_state['original_filename']}**")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Simulate bot response (replace with real processing)
    response = f"🤖 You asked: '{prompt}'.\n\n(Here’s where DB/LLM results go.)"

    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
