# file_utils.py
import os
import math
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st
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
import torch

# Ensure upload dir exists
os.makedirs(UPLOAD_FILE_DIR, exist_ok=True)

# Tokenizer parallelism
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")



# =========================
# OPTIMIZED: device pick
# =========================
def _file_key(uploaded_file) -> str:
    """Stable fingerprint for a file based on its raw bytes."""
    # read bytes without consuming the stream for later use
    data = uploaded_file.getvalue()  # Streamlit uploader supports this
    return hashlib.sha1(data).hexdigest()


def _is_inference_tensor_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "version_counter" in msg and "inference tensor" in msg
# =========================
# OPTIMIZED: device pick
# =========================
def _pick_device():
    # CUDA (also covers ROCm builds exposed via torch.cuda)
    if torch.cuda.is_available():
        # Nice free speed on Ampere+ for matmul (and optionally cuDNN)
        torch.backends.cuda.matmul.allow_tf32 = True  # https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
        try:
            torch.backends.cudnn.allow_tf32 = True   # https://pytorch.org/docs/stable/backends.html#torch-backends-cudnn-allow-tf32
        except Exception:
            pass
        return torch.device("cuda"), "cuda"

    # DirectML (Windows, any DX12 GPU)
    try:
        import torch_directml  # pip install torch-directml
        dml_dev = torch_directml.device()
        return dml_dev, "directml"  # https://learn.microsoft.com/windows/ai/directml/pytorch-windows
    except Exception:
        pass

    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"

    return torch.device("cpu"), "cpu"

DEVICE, DEVICE_KIND = _pick_device()
if DEVICE_KIND != "cuda":
    warnings.filterwarnings("ignore", message=".*pin_memory.*")  # see PyTorch warning text


if DEVICE_KIND == "cpu":
    try:
        torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
    except Exception:
        pass

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
    global _TOKENIZER
    if _TOKENIZER is None:
        if HUGGING_FACE_CACHING:
            os.makedirs(HUGGING_FACE_CACHING, exist_ok=True)
        _TOKENIZER = AutoTokenizer.from_pretrained(
            name,
            cache_dir=HUGGING_FACE_CACHING or None,
            model_max_length=1_000_000,
            truncation_side="right",
            use_fast=True,             # <- was False
        )
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
        chunker = HybridChunker(tokenizer=tokenizer, max_tokens=1024, merge_peers=True)

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

    # --- dynamic batch size ---
    def _bsz():
        if DEVICE_KIND == "cuda":
            return int(os.getenv("EMBED_BATCH", "128"))
        return int(os.getenv("EMBED_BATCH", "32"))     
        
    # Prefer multiple devices when available (v5+)
    devices = None
    if DEVICE_KIND == "cuda" and torch.cuda.device_count() > 1:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    def _encode(device_to_use):
        try:
            return model.encode(
                texts,
                device=(devices if (device_to_use == "auto" and devices) else (DEVICE if device_to_use == "auto" else device_to_use)),
                chunk_size=512 if (device_to_use == "auto" and devices and len(devices) > 1) else None,
                batch_size=_bsz(),
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except TypeError:
            # Older sentence-transformers fallback (no list-of-devices support)
            if device_to_use == "auto" and devices and len(devices) > 1:
                pool = model.start_multi_process_pool(devices)
                try:
                    return model.encode_multi_process(
                        texts, pool, chunk_size=512, batch_size=_bsz(),
                        normalize_embeddings=True, show_progress_bar=False
                    )
                finally:
                    model.stop_multi_process_pool(pool)
            return model.encode(
                texts, batch_size=_bsz(),
                normalize_embeddings=True, show_progress_bar=False,
                convert_to_numpy=True,
                device=(DEVICE if device_to_use == "auto" else device_to_use),
            )
    try:
        # First try: your selected accelerator (CUDA / MPS / DirectML / CPU)
        embeddings = _encode("auto")
    except RuntimeError as e:
        # Typical on DirectML (and sometimes other alt backends) due to inference tensors.
        if _is_inference_tensor_error(e) or DEVICE_KIND == "directml":
            logger.warning("⚠️ Falling back to CPU for embeddings due to backend inference-tensor error: %s", e)
            # Conservative CPU batch size; make sure we don't inherit DML tensors
            cpu_device = torch.device("cpu")
            # Optional: cap threads on CPU
            try:
                torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
            except Exception:
                pass
            embeddings = _encode(cpu_device)
        else:
            raise

    return texts, metadatas, embeddings.tolist()




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
