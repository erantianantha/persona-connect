"""
packages/rag/retrieve.py
Shared retrieval module used by both the voice (FastAPI) and
chat (Next.js via Python API) services.

Embedding model: all-MiniLM-L6-v2  (sentence-transformers, local, FREE)
Vector store:    Pinecone  (free tier)
Reranker:        cross-encoder/ms-marco-MiniLM-L-6-v2  (sentence-transformers, local, FREE)

Query pipeline:
  1. Embed query with all-MiniLM-L6-v2  (384-dim, runs locally)
  2. Fetch top_k * 2 candidates from Pinecone
  3. Cross-encoder rerank → return top_k results
"""

import os
import time
from typing import Any

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, runs locally — no API key needed
PINECONE_INDEX  = os.environ.get("PINECONE_INDEX", "persona")
PINECONE_DIM    = 384                   # must match all-MiniLM-L6-v2 output
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ──────────────────────────────────────────────
# Singletons (initialised once at import time)
# ──────────────────────────────────────────────
_embedder: SentenceTransformer | None = None
_pc_index: Any = None
_reranker: CrossEncoder | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("[RAG] Loading embedding model (first call — cached after this)...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_index():
    global _pc_index
    if _pc_index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        if PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=PINECONE_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
                )
            )
        _pc_index = pc.Index(PINECONE_INDEX)
    return _pc_index


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def retrieve(query: str, top_k: int = 4, fetch_k: int | None = None) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a given query.

    Returns a list of dicts:
        [{"text": str, "score": float, "source": str, "metadata": dict}, ...]
    """
    if not query or not query.strip():
        return []

    if fetch_k is None:
        fetch_k = max(top_k * 5, 30)

    t0 = time.time()

    # 1. Embed query locally (free, no API call)
    embedder = _get_embedder()
    query_vector = embedder.encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    )[0].tolist()

    t1 = time.time()
    print(f"[RAG] Embedding: {(t1-t0)*1000:.0f}ms")

    # 2. Vector search
    index = _get_index()
    results = index.query(
        vector=query_vector,
        top_k=fetch_k,
        include_metadata=True
    )

    matches = results.get("matches", [])
    if not matches:
        return []

    t2 = time.time()
    print(f"[RAG] Pinecone search ({len(matches)} hits): {(t2-t1)*1000:.0f}ms")

    # 3. Cross-encoder rerank (also local and free)
    texts = [m["metadata"].get("text", "") for m in matches]
    pairs = [(query, t) for t in texts]

    reranker = _get_reranker()
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(matches, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    t3 = time.time()
    print(f"[RAG] Rerank: {(t3-t2)*1000:.0f}ms  | Total RAG: {(t3-t0)*1000:.0f}ms")

    return [
        {
            "text":     match["metadata"].get("text", ""),
            "score":    float(score),
            "source":   match["metadata"].get("source", "unknown"),
            "metadata": {k: v for k, v in match["metadata"].items() if k != "text"},
        }
        for match, score in ranked
    ]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Used by the ingest pipeline. Runs locally — free."""
    embedder = _get_embedder()
    embeddings = embedder.encode(
        texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64
    )
    return embeddings.tolist()
