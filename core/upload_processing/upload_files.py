# file_utils.py
import os
import math
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st
import torch
from loguru import logger

from core.global_settings import UPLOAD_FILE_DIR, HUGGING_FACE_CACHING

# ---- Loaders / chunking ----
from langchain_community.document_loaders import CSVLoader
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

# ---- Embeddings ----
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import hashlib

#Chromadb client and collection
from core.utils.chromadb import _get_chroma_collection

# Ensure upload dir exists
os.makedirs(UPLOAD_FILE_DIR, exist_ok=True)


# =========================
# OPTIMIZED: device pick
# =========================
def _file_key(uploaded_file) -> str:
    """Stable fingerprint for a file based on its raw bytes."""
    # read bytes without consuming the stream for later use
    data = uploaded_file.getvalue()  # Streamlit uploader supports this
    return hashlib.sha1(data).hexdigest()

# =========================
# OPTIMIZED: device pick
# =========================
def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"

DEVICE = _pick_device()
if DEVICE == "cpu":
    # suppress harmless pin_memory warning when running on CPU
    warnings.filterwarnings("ignore", message=".*pin_memory.*")
logger.info(f"🚀 Using device: {DEVICE}")

# =========================
# OPTIMIZED: global caches
# =========================
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_TOKENIZER: AutoTokenizer | None = None

def _get_model(model_name: str = "nomic-ai/nomic-embed-text-v1.5") -> SentenceTransformer:
    """Singleton-ish loader for the embedding model (per session)."""
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=DEVICE,               # << key: binds DataLoader/device correctly
        )
        if DEVICE == "cpu":
            try:
                torch.set_num_threads(max(1, os.cpu_count() or 1))
            except Exception:
                pass
    return _MODEL_CACHE[model_name]

def _get_tokenizer(name: str = "bert-base-uncased") -> AutoTokenizer:
    """Singleton-ish loader for docling chunker tokenizer."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(name, cache_dir=HUGGING_FACE_CACHING,
            model_max_length=1_000_000,   # << allow long sequences safely
            truncation_side="right",      # << just in case any internal truncation is applied
            use_fast=False,                 )
    return _TOKENIZER

# =========================
# Metadata utilities
# =========================
def _to_builtin_scalar(v: Any):
    """Cast numpy scalars -> builtins; drop NaN/Inf; return None for invalid."""
    if v is None:
        return None
    if isinstance(v, (str, bool, int, float)):
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        fv = float(v)
        return fv if math.isfinite(fv) else None
    return None

def _scalarize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten to Chroma-safe scalars (no None), and derive page_no if possible."""
    out: Dict[str, Any] = {}
    if not isinstance(meta, dict):
        return out

    keep_keys = ("source", "mime_type", "row_start", "row_end", "section", "heading", "page_no", "chunk_index")
    for k in keep_keys:
        val = _to_builtin_scalar(meta.get(k))
        if val is not None:
            out[k] = val

    # derive page_no from Docling provenance if present
    try:
        items = meta.get("doc_items") or []
        if isinstance(items, list):
            for it in items:
                prov = (it or {}).get("prov") or []
                if isinstance(prov, list) and prov:
                    pg = (prov[0] or {}).get("page_no")
                    pg_cast = _to_builtin_scalar(pg)
                    if isinstance(pg_cast, int):
                        out.setdefault("page_no", pg_cast)
                        break
    except Exception:
        pass
    return out

def _validate_metadatas_for_chroma(metadatas: List[Dict[str, Any]]):
    """Raise a helpful error pinpointing the first invalid key/value."""
    for i, m in enumerate(metadatas):
        if not isinstance(m, dict):
            raise ValueError(f"metadata[{i}] is not a dict")
        for k, v in m.items():
            if v is None or isinstance(v, (list, dict, tuple, set)):
                raise ValueError(f"metadata[{i}]['{k}'] invalid (non-scalar or None): {v!r}")
            if not isinstance(v, (str, int, float, bool)):
                raise ValueError(f"metadata[{i}]['{k}'] non-scalar type {type(v).__name__}: {v!r}")
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"metadata[{i}]['{k}'] non-finite float: {v!r}")

# =========================
# Loading & Chunking
# =========================
def _load_as_documents(save_path: str, mime_type: str) -> List[Document]:
    """
    Load a file and return a list[Document] ready to embed.
    Handles CSV and PDF/DOCX via Docling.
    """
    if mime_type == "text/csv":
        loader = CSVLoader(file_path=save_path)
        row_docs = loader.load()

        # Merge rows (reduces chunk overhead) then split
        merged: List[Document] = []
        rows_per_chunk = 10
        chunk_size = 1000
        for i in range(0, len(row_docs), rows_per_chunk):
            batch = row_docs[i : i + rows_per_chunk]
            combined_text = "\n\n".join(r.page_content for r in batch)
            metadata = {
                "source": save_path,
                "row_start": i,
                "row_end": i + len(batch) - 1,
                "mime_type": "text/csv",
            }
            merged.append(Document(page_content=combined_text, metadata=metadata))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1),
        )
        return splitter.split_documents(merged)

    elif mime_type in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        converter = DocumentConverter()
        tokenizer = _get_tokenizer()  # cached
        chunker = HybridChunker(tokenizer=tokenizer, max_tokens=512, merge_peers=True)

        loader = DoclingLoader(
            file_path=[save_path],
            converter=converter,
            export_type=ExportType.DOC_CHUNKS,
            chunker=chunker,
        )
        docs = loader.load()
        for d in docs:
            d.metadata = {**(d.metadata or {}), "source": save_path, "mime_type": mime_type}
        return docs

    else:
        raise ValueError(f"Unsupported file type: {mime_type}")

# =========================
# Embedding
# =========================
def _embed_texts(docs: List[Document]) -> Tuple[List[str], List[Dict[str, Any]], List[List[float]]]:
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata or {} for d in docs]
    if not texts:
        return [], [], []

    model = _get_model()

    batch_size = 128 if DEVICE == "cuda" else (64 if DEVICE == "mps" else 32)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    return texts, metadatas, embeddings



# =========================
# Public API
# =========================
def process_uploaded_file(uploaded_file) -> Tuple[str, str]:
    """Save file, process (CSV/PDF/DOCX), embed, and store in Chroma."""
    # Ensure session state key exists (safe on reruns)
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    file_key = _file_key(uploaded_file)

    save_path = os.path.join(UPLOAD_FILE_DIR, f"{file_key}_{uploaded_file.name}")
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.uploaded_files.append(save_path)
    mime_type = uploaded_file.type

    client, collection = _get_chroma_collection()

    existing = collection.get(where={"file_key": file_key},  limit=1)
    if existing and existing.get("ids"):
        logger.info(f"🔁 Reusing cached index for file_key={file_key}")
        # Update session so the chat page knows which file to query
        st.session_state["original_filename"] = uploaded_file.name
        st.session_state["last_file_id"] = file_key
        return "Pass", uploaded_file.name

    logger.info("📄 Document processing started")
    docs = _load_as_documents(save_path, mime_type)

    logger.info("🧮 Creating embeddings")
    texts, metadatas, embeddings = _embed_texts(docs)
    if not texts:
        return "No content to index", mime_type

    # # sanitize + enrich metadata
    metadatas = [_scalarize_metadata(m) for m in metadatas]
    for i, m in enumerate(metadatas):
        m["chunk_index"] = i
        m["file_key"] = file_key

    _validate_metadatas_for_chroma(metadatas)

    #  build persistent IDs from file_key (not session-specific file_id)
    ids = [f"{file_key}_doc_{i}" for i in range(len(texts))]

    

    # base = str(uploaded_file.file_id)
    # ids = [f"{base}_doc_{i}" for i in range(len(texts))]

    for i, mid in enumerate(ids):
        metadatas[i]["id"] = mid

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    try:
        client.persist()
    except Exception:
        pass

    logger.info("✅ Data added to vector DB. Ready for retrieval.")
    st.session_state["original_filename"] = uploaded_file.name
    st.session_state["last_file_id"] = file_key
    return "Pass", uploaded_file.name

    # return "Pass", "pass"

# def cleanup_uploaded_files():
#     """Remove session-tracked files safely."""
#     if "uploaded_files" not in st.session_state:
#         return
#     for path in list(st.session_state.uploaded_files):
#         try:
#             if os.path.isfile(path):
#                 os.remove(path)
#         except Exception:
#             pass
#     st.session_state.uploaded_files.clear()
