# core/llm/Retrivals/text_embedding_retrival.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document as LCDocument
from langchain.embeddings.base import Embeddings
from core.upload_processing.upload_files import _get_model

from core.utils.chromadb import _get_chroma_collection


try:
    from FlagEmbedding import FlagReranker
    _HAS_RERANKER = True
except Exception:
    _HAS_RERANKER = False


# -------------------------
# Embedding adapter (LC API)
# -------------------------
class SentenceTransformerEmbeddings(Embeddings):
    """LangChain Embeddings wrapper over your _get_model()."""
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model = _get_model(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()


# -------------------------
# Vector store binding
# -------------------------
def _get_vectorstore() -> Chroma:
    """Bind LangChain's Chroma to your existing chromadb client/collection."""
    client, _ = _get_chroma_collection()
    emb = SentenceTransformerEmbeddings()
    # IMPORTANT: pass the existing client and the known collection name
    vs = Chroma(client=client, collection_name="knowledge_base", embedding_function=emb)
    return vs


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



def _build_bm25_for_file(file_id: Optional[str]) -> BM25Retriever:
    """
    BM25 scores a document by:

    Term frequency (tf): more occurrences of a query term → higher score (with diminishing returns).

    Inverse document frequency (idf): rarer terms → higher weight.

    Length normalization (b): longer docs don’t get unfair advantage.
    Two main hyperparams:

    k1 (typ. 1.2–2.0): tf saturation.

    b (0–1): length normalization strength.
    """
    texts, metadatas, _ = _fetch_texts_metas_ids_for_file(file_id)
    bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas)
    bm25.k = 25
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
    if use_mmr:
        docs = vs.max_marginal_relevance_search(
            query, k=top_k, fetch_k=max(top_k * 3, 120), lambda_mult=mmr_lambda
        )
    else:
        docs = vs.similarity_search(query, k=top_k)

    if not file_id:
        return docs

    prefix = f"{file_id}_doc_"
    out: List[LCDocument] = []
    for d in docs:
        m = d.metadata or {}
        mid = m.get("id") or m.get("_id") or ""
        if str(mid).startswith(prefix):
            out.append(d)
    return out

_RERANKER = None
def _init_reranker():
    global _RERANKER
    if not _HAS_RERANKER:
        return None
    if _RERANKER is None:
        _RERANKER = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    return _RERANKER


def _apply_rerank(reranker, query: str, docs: List[LCDocument], top_n: int) -> List[LCDocument]:
    if not reranker or not docs:
        return docs[:top_n]
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.compute_score(pairs)  # higher is better
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]


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
    lines = [f"**Answer context for:** `{query}`\n"]
    for i, d in enumerate(docs, 1):
        m = d.metadata or {}
        src = os.path.basename(m.get("source", "unknown"))
        page = m.get("page_no", m.get("row_start", "—"))
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > 500:
            snippet = snippet[:500] + "…"
        lines.append(f"**{i}.** p. {page} — `{src}`")
        lines.append(f"> {snippet}\n")
    lines.append("_(Tip: feed these as context passages into your LLM.)_")
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

    bm25 = _build_bm25_for_file(file_id)
    bm25_docs = bm25.invoke(query)

    fused = _dedupe_keep_order(bm25_docs + dense_docs)

    reranker = _init_reranker()
    top_docs = _apply_rerank(reranker, query, fused, top_n=rerank_top_n)

    return _pretty_answer(query, top_docs)
