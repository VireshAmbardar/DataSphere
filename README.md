

# DataSphere

A **Streamlit**‑based document Q\&A assistant that lets users upload files (PDF, DOCX, CSV), build embeddings, and ask natural‑language questions. It powers a **Retrieval‑Augmented Generation (RAG)** pipeline using **ChromaDB**, **LangChain**, and **Groq** LLMs.

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Technical Architecture](#technical-architecture)

   * File Handling & Upload
   * Embedding Generation
   * Vector Store & Retrieval
   * Reranking & Fusion
   * LLM Response with Groq
4. [Development Setup](#development-setup)
5. [Usage](#usage)
6. [Extending & Customizing](#extending--customizing)

---

## Overview

**DataSphere** allows users to:

* Upload a document (PDF, Word, CSV)
* Extract Texts and process its content into chunks 
* Generate semantic embeddings to index in a vector database
* Ask questions—retrieval returns relevant chunks followed by an LLM-formulated response

It’s intuitive: choose your file, wait for processing (with an improved loading spinner UX), and then ask away!

---

## How It Works

1. **User Upload**: The user selects a file and uploads it via a Streamlit interface.
2. **Processing**: The file is parsed, chunked, and embedded — all wrapped in a spinner for loading feedback.
3. **Indexing**: Embeddings are stored in **ChromaDB** with metadata, enabling fast similarity search.
4. **Query & Retrieve**: When the user asks a question, the system retrieves top chunks via dense (Chroma) and sparse (BM25) methods.
5. **Reranking & Deduplication**: Retrieved chunks are fused, reranked with scoring, and duplicates are removed to craft an ordered list.
6. **LLM Response**: Relevant passages are passed to a Groq-powered LLM endpoint, which generates a polished, human-readable answer.

---

## 🚀 Upcoming Updates

- **Expanded Data Sources**  
  Currently, only textual content from PDF and DOCX files is processed.  
  Future plans include support for **SQL databases** and **Postgres connections**, enabling direct querying of structured data sources.

- **UI Refinements**  
  Ongoing improvements in the **Streamlit UI** for a more intuitive and polished user experience.

- **Advanced Content Extraction**  
  Work is in progress to extract and embed not just plain text, but also **entities such as images and tables** from documents, making responses more context-rich.



## Technical Architecture


* **Parsing Logic**:

  * CSV: loaded via `CSVLoader`, grouped, and chunked using `RecursiveCharacterTextSplitter`.
  * PDF/DOCX: handled by `DoclingLoader` with `HybridChunker`.

* **Embeddings Pipeline**:

  * Singleton `SentenceTransformer` model via `_get_model`.
  * Dynamically selects device (GPU/CPU/DirectML/MPS) via `_pick_device`.
  * Supports multi-device parallel embedding when CUDA is available; fallback to CPU on edge-case errors.

* **Indexing**:

  * Chunks are embedded, metadata sanitized, and stored in ChromaDB.
  * Sessions store `file_key`, `original_filename`, and `processed_file` for retrieval.

### Vector Store & Retrieval (`text_embedding_retrival.py`)

* **Embedding Adapter**: Wraps your model for LangChain compatibility, handling batching, multi-device encoding, and fallback.

* **Vector Store**: Initializes a `Chroma` store using your existing ChromaDB client. Cached for repeated use.

* **Retrieval Methods**:

  1. **Dense Retrieval**: Uses Chroma’s similarity search or MMR-based search for diversity.
  2. **Sparse Retrieval (BM25)**: Uses BM25Retriever from LangChain to score based on term frequency and document structure.

* **Fusion & Ranking**:

  * Combines dense + sparse results, removes duplicates while preserving order.
  * Optionally reranks using `FlagReranker`, with fallback to CPU on tensor errors.

* **Answer Formatting**:

  * Top documents are converted into snippet strings and passed to your LLM generator, which replies with a structured answer.

### LLM Response Generation (`groq_connector` module)

* Defines prompts/background context to guide Groq-based LLM responses.
* Constructs a system/user message pair, streaming response via Groq's SDK interface.
* Highlights are formatted in Markdown with bulleted or numbered clarity.

---

## Development Setup

1. **Python** ≥ 3.12.10
2. Install dependencies:

   ```bash
   poetry install
   ```
3. Ensure you have a Groq API key in `.env`:

   ```
   GROQ_API_KEY=your_api_key_here
   ```
4. Run the app:

   ```bash
   streamlit run apps/streamlit/DataSphere.py --server.port 8501
   ```

---

## Usage

1. Visit the app in your browser (usually at `localhost:8501`).
2. Upload a supported file (PDF, DOCX, or CSV).
3. Wait for processing (spinner shows during embedding & vector store creation).
4. Once it completes, switch to the Q\&A page automatically.
5. Ask your question, get a refined answer leveraging document context.

---

## Extending & Customizing

* **Add More File Formats**: Extend `_load_as_documents` to support TXT, HTML, etc.
* **Alternate Embeddings**: Swap out `nomic-ai/nomic-embed-text-v1.5` for OpenAI or other embedder.
* **Alternate Vector Stores**: Replace Chroma with Pinecone, Weaviate, or others.
* **LLM Variants**: Replace Groq with OpenAI, Claude, or local LLMs with minimal adaptation.
* **UI Enhancements**: Add features like query history, feedback tools, or PDF preview.

---

## Learn More

This project is inspired by RAG applications built in Streamlit with LangChain and ChromaDB. Such techniques are increasingly popular for building **semantic search and document Q\&A** tools.

See guides like *“Building an Interactive Document Q\&A App with Streamlit and LangChain”* for in-depth walkthroughs.([drlee.io][1], [medium.com][2], [ai.gopubby.com][3], [medium.com][4], [python.plainenglish.io][5], [medium.com][6])

---

## Summary

**DataSphere** is a lightweight, end-to-end retrieval-augmented Q\&A system that unifies file upload, embedding generation, vector indexing, hybrid retrieval, reranking, and LLM answer generation—wrapped in a clean Streamlit interface with polished UX enhancements like spinners and page navigation.

Feel free to adjust, extend, or integrate this into your own AI‐powered documentation tools!