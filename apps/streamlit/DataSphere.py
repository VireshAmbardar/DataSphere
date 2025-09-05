import os
import sys
import streamlit as st

try:
    import pysqlite3, sys
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

# make repo root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.upload_processing.upload_files import process_uploaded_file
# from core.upload_processing.upload_files import cleanup_uploaded_files  # keep if you want a manual "clear" button

# ────────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────────
st.title("Welcome to _DataSphere_:blue")

st.markdown(
    """
    <h3 style='text-align: center; font-size: 24px;'>
        Welcome to <b>AskMyDB</b> — a simple tool where you can Upload Your local File and, 
        ask natural language questions, and generate insights.
    </h3>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h4 style='text-align: center; color: gray;'>Choose Your Data Source 📂</h4>",
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────────
# Session defaults (kept minimal; process_uploaded_file will set these too)
# ────────────────────────────────────────────────────────────────────────────────
st.session_state.setdefault("selected_source", None)
st.session_state.setdefault("last_file_id", None)   
st.session_state.setdefault("original_filename", None)
st.session_state.setdefault("processed_file", None)
st.session_state.setdefault("process_state", None)

# ────────────────────────────────────────────────────────────────────────────────
# Source selection
# ────────────────────────────────────────────────────────────────────────────────
items = [
    ("PDF", ":material/picture_as_pdf:"),
    ("DOCX", ":material/docs:"),
    ("CSV", ":material/csv:"),
    # ("SQL", ":material/database:"),
    # ("Postgres", ":material/database:"),
]

cols = st.columns(len(items))
for col, (label, icon) in zip(cols, items):
    with col:
        if st.button(label, key=f"src_{label}", icon=icon, use_container_width=True):
            st.session_state["selected_source"] = label

# ────────────────────────────────────────────────────────────────────────────────
# Upload widgets for file-backed sources
# ────────────────────────────────────────────────────────────────────────────────
selected = st.session_state.get("selected_source")
UPLOAD_MAP = {
    "PDF":  (["pdf"],  "Upload PDF"),
    "DOCX": (["docx"], "Upload DOCX"),
    "CSV":  (["csv"],  "Upload CSV"),
}

if selected in UPLOAD_MAP:
    exts, prompt = UPLOAD_MAP[selected]

    uploaded_file = st.file_uploader(f"{prompt}", type=exts)
    if uploaded_file:
        # Always call process_uploaded_file — it will:
        # 1) compute a stable file_key from bytes
        # 2) check Chroma for existing chunks (where={"file_key": file_key})
        # 3) skip parsing/embedding if already cached
        # 4) set session: original_filename, last_file_id (file_key)
        with st.spinner("Processing your file… this may take a moment"):
            st.session_state["process_state"] = "processing"
            status, fname = process_uploaded_file(uploaded_file)
        
        if status == "Pass":
            st.session_state["processed_file"] = fname  # human-friendly name
            st.session_state["process_state"] = "done"
            st.toast("Indexed successfully ✅")
            # navigate to chat page
            st.switch_page("pages/ask_questions.py")
        else:
            st.session_state["process_state"] = "error"
            st.error(f"Failed to process file: {status} ({fname})")

else:
    st.info("Pick a data source above to begin.")

# ────────────────────────────────────────────────────────────────────────────────
# Optional: quick status block
# ────────────────────────────────────────────────────────────────────────────────
# with st.expander("Current session status", expanded=False):
#     st.write({
#         "selected_source": st.session_state.get("selected_source"),
#         "original_filename": st.session_state.get("original_filename"),
#         "last_file_id (file_key)": st.session_state.get("last_file_id"),
#         "processed_file": st.session_state.get("processed_file"),
#     })

# ────────────────────────────────────────────────────────────────────────────────
# Optional: manual cleanup button (if you implemented cleanup_uploaded_files)
# ────────────────────────────────────────────────────────────────────────────────
# if st.button("Clear uploaded file cache from disk", type="secondary"):
#     cleanup_uploaded_files()
#     st.success("Local uploaded files cleaned from disk.")
