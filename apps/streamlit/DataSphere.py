import streamlit as st
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from core.upload_processing.upload_files import process_uploaded_file,cleanup_uploaded_files

# ✅ Clear cache safely
st.cache_data.clear()
st.cache_resource.clear()

# ✅ App title
st.title("Welcome to _DataSphere_ is :blue[cool] :sunglasses:")

# ✅ Description
st.markdown(
    """
    <h3 style='text-align: center; font-size: 24px;'>
        Welcome to <b>AskMyDB</b> — a simple tool where you can connect your database, 
        ask natural language questions, and generate insightful reports with charts 📊.<br>
        Use the sidebar to navigate between <b>Connect</b>, <b>Ask</b>, <b>Reports</b>, and <b>Settings</b>.
    </h3>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h4 style='text-align: center; color: gray;'>Choose Your Data Source 📂</h4>",
    unsafe_allow_html=True,
)

# ✅ Data sources
items = [
    ("PDF", ":material/picture_as_pdf:"),
    ("DOCX", ":material/docs:"),
    ("CSV", ":material/csv:"),
    ("SQL", ":material/database:"),
    ("Postgres", ":material/database:"),
]

# ✅ Initialize session_state keys early
for k, v in {
    "selected_source": None,
    "last_file_id": None,
    "processed_file": None,
}.items():
    st.session_state.setdefault(k, v)

cols_per_row = 3

# ✅ Render buttons in grid layout
# Source selection
cols = st.columns(len(items))
for col, (label, icon) in zip(cols, items):
    with col:
        if st.button(label, key=f"src_{label}", icon=icon, use_container_width=True):
            st.session_state["selected_source"] = label

# ✅ Handle selected source safely
selected = st.session_state.get("selected_source")
UPLOAD_MAP = {
    "PDF":  (["pdf"],  "Upload PDF"),
    "DOCX": (["docx"], "Upload DOCX"),
    "CSV":  (["csv"],  "Upload CSV"),
}
if selected in UPLOAD_MAP:
    exts, prompt = UPLOAD_MAP[selected]

    # take file from user
    uploaded_file = st.file_uploader(f"Upload {prompt}", type=exts)


    if uploaded_file:
        # get file id
        current_file_id = getattr(uploaded_file, "file_id", f"{uploaded_file.name}:{uploaded_file.size}")

        # Only process if new file
        if st.session_state.get("last_file_id") != current_file_id:
            st.session_state["last_file_id"] = current_file_id

            # Clean up old uploads
            cleanup_uploaded_files()
       
            status, file_name = process_uploaded_file(uploaded_file)
            if status == "Pass":
                st.session_state["processed_file"] = file_name
                # print("Enjoy")   # will run only once

                # redirect to Ask Questions page
                st.switch_page("pages/ask_questions.py")
            else:
                st.error(f"Failed to process file: {status} ({file_name})")


        else:
            st.info(f"Already processed: {st.session_state['processed_file']}")


