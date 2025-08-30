# file_utils.py
import os
import shutil
from uuid import UUID
import streamlit as st
from core.global_settings import UPLOAD_FILE_DIR,HUGGING_FACE_CACHING,CHROMADB_CACHE_DIR

from langchain_community.document_loaders import CSVLoader
from langchain_docling import DoclingLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_docling.loader  import ExportType
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from langchain_core.documents import Document
import chromadb
from chromadb.config import Settings


os.makedirs(UPLOAD_FILE_DIR, exist_ok=True)

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

def process_uploaded_file(uploaded_file):
    """Save file, process pdf/docx/csv, and return LlamaIndex Document(s)."""
    save_path = os.path.join(UPLOAD_FILE_DIR, f"{uploaded_file.file_id}_{uploaded_file.name}")
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    

    # save path in state
    st.session_state.uploaded_files.append(save_path)
    st.session_state.uploaded_files.append(UPLOAD_FILE_DIR)
    mime_type = uploaded_file.type

    #-----------------
    # First comes Making document
    #-----------------

    # ----- CSV -----
    if mime_type == "text/csv":
        loader = CSVLoader(file_path=save_path)
        docs = loader.load()

        merged = []
        rows_per_chunk = 10
        chunk_size =1000
        for i in range(0, len(docs), rows_per_chunk):
            batch = docs[i : i + rows_per_chunk]
            combined_text = "\n\n".join(doc.page_content for doc in batch)
            metadata = {
                "source": save_path,
                "row_start": i,
                "row_end": i + len(batch) - 1
            }
            merged.append(Document(page_content=combined_text, metadata=metadata))
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.1))
        document = splitter.split_documents(merged)

    # ----- PDF/DOCS -----
    elif mime_type in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        # loader = DoclingLoader(file_path=[save_path], export_type=ExportType.DOC_CHUNKS)
        # documents = loader.load()
        converter = DocumentConverter()
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",cache_dir=HUGGING_FACE_CACHING)
        chunker = HybridChunker(tokenizer=tokenizer, max_tokens=512, merge_peers=True)

        loader = DoclingLoader(
            file_path=[save_path],
            converter=converter,
            export_type=ExportType.DOC_CHUNKS,
            chunker=chunker
        )
        document = loader.load()


    # print(document)

    #-----------------
    # Making Enbeddings from document
    #-----------------
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    embeddings = model.encode(document)

    # Chrome DB client
    client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=CHROMADB_CACHE_DIR
    ))

    # make collection
    collection = client.get_or_create_collection(name="knowledge_base")

    # add data to chromadb
    collection.add(
    documents=document,
    embeddings=embeddings,
    metadatas=[{"source": f"doc_{i}"} for i in range(len(document))],
    ids=[f"doc_{i}" for i in range(len(document))]
    )
    


    return "Pass", "pass"


def cleanup_uploaded_files():
    """Remove session-tracked files."""
    for path in st.session_state.get("uploaded_files", []):
        if os.path.exists(path):
            os.remove(path)
    st.session_state.uploaded_files.clear()
