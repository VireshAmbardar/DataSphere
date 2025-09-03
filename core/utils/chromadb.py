from core.global_settings import CHROMADB_CACHE_DIR

# ---- Vector store ----
import chromadb

# =========================
# Chroma collection
# =========================
def _get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMADB_CACHE_DIR)
    collection = client.get_or_create_collection(name="knowledge_base")
    return client, collection