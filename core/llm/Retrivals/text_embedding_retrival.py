# core/llm/Retrivals/text_embedding_retrival.py
from __future__ import annotations

from functools import lru_cache
import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document as LCDocument
from langchain.embeddings.base import Embeddings
from core.llm.LLMConnector.groq_connector import response_generator
from core.upload_processing.upload_files import _get_model,_pick_device

from core.utils.chromadb import _get_chroma_collection
from FlagEmbedding import FlagReranker
import torch
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# -------------------------
# Device normalize (works whether _pick_device returns a torch.device,
# a (device, kind) tuple, a string, or a DirectML device object)
# -------------------------
def _normalize_device(dev):
    # Tuple case: (torch.device or dml_device, "kind")
    if isinstance(dev, tuple) and len(dev) == 2:
        return dev[0], dev[1]

    # torch.device
    if hasattr(dev, "type"):  # cuda / cpu / mps
        kind = getattr(dev, "type", "unknown")
        return dev, kind

    # string ("cuda"/"cpu"/"mps")
    if isinstance(dev, str):
        if dev in {"cuda", "cpu", "mps"}:
            return torch.device(dev), dev
        # unknown string, assume cpu
        return torch.device("cpu"), "cpu"

    # likely DirectML device object
    mod = getattr(dev.__class__, "__module__", "")
    name = getattr(dev.__class__, "__name__", "")
    if "torch_directml" in mod or "DirectML" in name:
        return dev, "directml"

    return torch.device("cpu"), "cpu"

DEVICE, DEVICE_KIND = _normalize_device(_pick_device())

# CUDA free speed (Ampere+): TF32 matmul/cudnn
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

# -------------------------
# Helpers for backend fallbacks
# -------------------------
def _is_inference_tensor_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "version_counter" in msg and "inference tensor" in msg

# -------------------------
# Embedding adapter (LC API)
# -------------------------
class SentenceTransformerEmbeddings(Embeddings):
    """LangChain Embeddings wrapper over your cached _get_model().
       - v5 multi-device encode when available
       - graceful CPU retry for DirectML/inference-tensor issues
    """
    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        batch_size: Optional[int] = None,
    ):
        self.model = _get_model(model_name)

        # batch size knobs
        if batch_size is not None:
            self._bsz = batch_size
        else:
            env_bsz = os.getenv("EMBED_BATCH")
            if env_bsz:
                self._bsz = int(env_bsz)
            elif DEVICE_KIND == "cuda":
                self._bsz = 192  # tune 128–256 by VRAM
            else:
                self._bsz = 32   # mps/directml/cpu default

        # detect Streamlit Cloud
        self.is_cloud = os.environ.get("STREAMLIT_RUNTIME") == "1"

        # Multi-GPU only for CUDA
        self._devices: Optional[List[str]] = None
        if DEVICE_KIND == "cuda" and torch.cuda.device_count() > 1:
            self._devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        elif DEVICE_KIND == "cpu":
            if self.is_cloud:
                # Streamlit Cloud → keep it safe
                self._devices = ["cpu"]
            else:
                workers = min(8, max(2, (os.cpu_count() or 2) // 2))
                self._devices = ["cpu"] * workers

    def _encode_once(self, texts: List[str], device_spec) -> List[List[float]]:
        """One attempt to encode on a given device (or 'auto' to use configured)."""
        # SentenceTransformers v5+ accepts device=str|list[str]|torch.device
        try:
            embeddings = self.model.encode(
                texts,
                device=(self._devices if (device_spec == "auto" and self._devices)
                        else (DEVICE if device_spec == "auto" else device_spec)),
                chunk_size=512 if (device_spec == "auto" and self._devices and len(self._devices) > 1) else None,
                batch_size=self._bsz,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except TypeError:
            # Older ST fallback: explicit multi-process pool
            if device_spec == "auto" and self._devices and len(self._devices) > 1:
                pool = self.model.start_multi_process_pool(self._devices)
                try:
                    embeddings = self.model.encode_multi_process(
                        texts, pool, chunk_size=512, batch_size=self._bsz,
                        normalize_embeddings=True, show_progress_bar=False
                    )
                finally:
                    self.model.stop_multi_process_pool(pool)
            else:
                embeddings = self.model.encode(
                    texts,
                    device=(DEVICE if device_spec == "auto" else device_spec),
                    batch_size=self._bsz,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
        return embeddings.tolist()

    def _encode(self, texts: List[str]) -> List[List[float]]:
        # 1) try configured device(s)
        try:
            return self._encode_once(texts, "auto")
        except RuntimeError as e:
            # Typical on alt backends (esp. DirectML) → retry on CPU
            if _is_inference_tensor_error(e) or DEVICE_KIND == "directml":
                # ensure sane CPU threading
                try:
                    torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
                except Exception:
                    pass
                return self._encode_once(texts, torch.device("cpu"))
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        return self._encode([text])[0]



# -------------------------
# Vector store binding
# -------------------------
@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    """Bind LangChain's Chroma to your existing chromadb client/collection."""
    client, _ = _get_chroma_collection()
    emb = SentenceTransformerEmbeddings()
    # IMPORTANT: pass the existing client and the known collection name
    return Chroma(client=client, collection_name="knowledge_base", embedding_function=emb)


# -------------------------
# Helpers
# -------------------------
def _fetch_texts_metas_ids_for_file(file_id: Optional[str]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Pull all chunks for a given file_key (passed in via file_id) using a metadata equality filter.
    We no longer prefix-scan IDs; we query by where={"file_key": file_id}.
    """
    client, collection = _get_chroma_collection()

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    ids: List[str] = []

    # If no file_id given, fall back to fetching everything (not typical in your flow).
    if not file_id:
        res = collection.get(include=["documents", "metadatas"])  # IDs always included implicitly
        ids = res.get("ids") or []
        texts = res.get("documents") or []
        metas = res.get("metadatas") or []
    else:
        # Page through results for robustness on large docs
        offset = 0
        batch = 1000
        where = {"file_key": file_id}

        while True:
            res = collection.get(where=where, include=["documents", "metadatas"], limit=batch, offset=offset)
            chunk_ids = res.get("ids") or []
            if not chunk_ids:
                break

            r_docs = res.get("documents") or []
            r_metas = res.get("metadatas") or []

            ids.extend(chunk_ids)
            texts.extend(r_docs)
            metas.extend(r_metas)

            if len(chunk_ids) < batch:
                break
            offset += batch

    # Ensure we keep a copy of the physical id in metadata for dedupe/citations
    for i, _id in enumerate(ids):
        metas[i] = {**(metas[i] or {}), "_id": _id}

    return texts, metas, ids


_BM25_CACHE: Dict[str, BM25Retriever] = {}

def _bm25_preprocess(text: str) -> str:
    # simple, fast normalization for IDF/TF
    return re.sub(r"\W+", " ", text).lower().strip()

def _build_bm25_for_file(file_id: Optional[str],k_for_search: int) -> BM25Retriever:
    """
    BM25 scores a document by:

    Term frequency (tf): more occurrences of a query term → higher score (with diminishing returns).

    Inverse document frequency (idf): rarer terms → higher weight.

    Length normalization (b): longer docs don't get unfair advantage.
    Two main hyperparams:

    k1 (typ. 1.2-2.0): tf saturation.

    b (0-1): length normalization strength.

    Build (and cache) a BM25 retriever on the doc subset for a file_id.
    Tuned params: k1 ~1.5-1.8, b ~0.75. We fetch more than top_k to help fusion.
    """
    cache_key = f"{file_id or '*'}::{k_for_search}"
    if cache_key in _BM25_CACHE:
        bm25 = _BM25_CACHE[cache_key]
        bm25.k = k_for_search
        return bm25

    texts, metadatas, _ = _fetch_texts_metas_ids_for_file(file_id)
    bm25 = BM25Retriever.from_texts(
        texts,
        metadatas=metadatas,
        bm25_params={"k1": 1.6, "b": 0.75},
        preprocess_func=_bm25_preprocess,
    )
    bm25.k = k_for_search
    _BM25_CACHE[cache_key] = bm25
    return bm25


def _dense_candidates(
    vs: Chroma,
    query: str,
    *,
    top_k: int,
    use_mmr: bool,
    mmr_lambda: float,
    file_id: Optional[str],
) -> List[LCDocument]:
    """Dense search, then filter to a specific file_id (via metadata.id/_id prefix) if given.
    returns the top-k most similar docs to the query vector. Fast and simple; can be redundant (many near-duplicates).
    max_marginal_relevance_search (MMR): first gathers a larger pool fetch_k (e.g., max(top_k*3, 40)), then greedily selects k docs that balance relevance and diversity:

    Relevance = similarity(doc, query)

    Diversity = dissimilarity(doc, already_selected_docs)

    Trade-off is controlled by lambda_mult (λ):

    λ close to 1.0 → prioritize relevance

    λ close to 0.0 → prioritize diversity
    This helps avoid “samey” chunks and usually improves downstream answer quality.
    """
    _filter = {"file_key": file_id} if file_id else None
    if use_mmr:
        # fetch a larger pool for MMR, then diversify
        return vs.max_marginal_relevance_search(
            query, k=top_k, fetch_k=max(top_k * 3, 120), lambda_mult=mmr_lambda, filter=_filter
        )
    return vs.similarity_search(query, k=top_k, filter=_filter)

    

# -------------------------
# Reranker (singleton)
# -------------------------
_RERANKER = None
def _get_reranker() -> FlagReranker:
    global _RERANKER
    if _RERANKER is None:
        model_name = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        use_fp16 = os.getenv("RERANK_FP16", "1") != "0"
        _RERANKER = FlagReranker(model_name, use_fp16=use_fp16)
    return _RERANKER

def _safe_rerank_scores(reranker: FlagReranker, pairs: List[Tuple[str, str]], batch_size: int) -> List[float]:
    try:
        return reranker.compute_score(pairs, batch_size=batch_size)
    except RuntimeError as e:
        # If a backend trips over inference tensors, retry on CPU by
        # clearing CUDA visibility for this process (best-effort) and re-instantiating
        if _is_inference_tensor_error(e) or DEVICE_KIND == "directml":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hint CPU
            rr = FlagReranker(os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"), use_fp16=False)
            return rr.compute_score(pairs, batch_size=batch_size)
        raise


def _apply_rerank(reranker: FlagReranker, query: str, docs: List[LCDocument], top_n: int) -> List[LCDocument]:
    if not reranker or not docs:
        return docs[:top_n]
    pairs = [(query, d.page_content) for d in docs]
    batch_size = int(os.getenv("RERANK_BATCH", "64"))
    # FlagReranker supports batched compute_score(list_of_pairs, batch_size=...)
    scores = reranker.compute_score(pairs, batch_size=batch_size)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]


# -------------------------
# Dedupe
# -------------------------
def _dedupe_keep_order(docs: List[LCDocument]) -> List[LCDocument]:
    """
    Remove duplicate (or near-duplicate) documents while preserving the original order.
    It walks the list once, building a seen set of “keys” it has already emitted.
    """
    seen = set()
    out: List[LCDocument] = []
    for d in docs:
        m = d.metadata or {}
        did = m.get("_id") or m.get("id") or ""
        key = (did, d.page_content[:72])  # small content key to avoid exact dupes
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _pretty_answer(query: str, docs: List[LCDocument]) -> str:
    if not docs:
        return f"🤖 I didn’t find anything relevant for: **{query}**."
    lines: List[str] = []
    for d in docs:
        snippet = d.page_content.strip().replace("\n", " ")
        lines.append(f"> {snippet}\n")
    return "\n".join(lines)

# -------------------------
# Public API
# -------------------------
def chroma_retrieve(
    query: str,
    *,
    top_k: int = 20,
    use_mmr: bool = True,
    mmr_lambda: float = 0.7,
    file_id: Optional[str] = None,   # <- supported
    rerank_top_n: int = 8,
) -> str:
    """
    Hybrid retrieval over your 'knowledge_base' collection:
      - Dense (Chroma) + Sparse (BM25) fusion
      - Optional rerank (BAAI/bge-reranker-v2-m3, if installed)
      - Returns a markdown answer with citations/snippets
    """
    vs = _get_vectorstore()

    dense_docs = _dense_candidates(
        vs, query, top_k=top_k, use_mmr=use_mmr, mmr_lambda=mmr_lambda, file_id=file_id
    )

    bm25_k = max(3 * rerank_top_n, top_k)
    bm25 = _build_bm25_for_file(file_id,bm25_k)
    bm25_docs = bm25.invoke(query)

    fused = _dedupe_keep_order(bm25_docs + dense_docs)

    reranker = _get_reranker()
    pairs = [(query, d.page_content) for d in fused]
    batch_size = int(os.getenv("RERANK_BATCH", "64"))
    scores = _safe_rerank_scores(reranker, pairs, batch_size=batch_size)

    ranked = sorted(zip(fused, scores), key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in ranked[:rerank_top_n]]

    return response_generator(
        user_query = query,
        messages = _pretty_answer(query, top_docs)
    )