"""
packages/rag/retrieve.py
Shared retrieval module used by both the voice (FastAPI) and
chat (Next.js via Python API) services.

Embedding model: models/gemini-embedding-2 (via Google API - 384 dimensions)
Vector store:    Pinecone  (free tier)

This version is optimized for serverless/constrained RAM environments (like Render Free Tier).
It uses HTTP API calls for embeddings instead of loading heavy local PyTorch/transformers.
"""

import os
import time
from typing import Any
import httpx
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
PINECONE_INDEX  = os.environ.get("PINECONE_INDEX", "persona")
PINECONE_DIM    = 384                   # must match Pinecone index dimension
EMBEDDING_MODEL = "models/gemini-embedding-2"

# ──────────────────────────────────────────────
# Singletons
# ──────────────────────────────────────────────
_pc_index: Any = None

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

def _call_gemini_embed(texts: list[str]) -> list[list[float]]:
    api_key = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_GENERATIVE_AI_API_KEY environment variable is not set")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/{EMBEDDING_MODEL}:batchEmbedContents?key={api_key}"
    
    requests = []
    for text in texts:
        requests.append({
            "model": EMBEDDING_MODEL,
            "content": {
                "parts": [{"text": text}]
            },
            "outputDimensionality": PINECONE_DIM
        })
    
    payload = {"requests": requests}
    
    max_retries = 5
    backoff = 3.0
    
    with httpx.Client(timeout=30.0) as client:
        for attempt in range(max_retries):
            response = client.post(url, json=payload)
            if response.status_code == 429:
                print(f"[RAG] 429 rate limit hit. Retrying in {backoff:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(backoff)
                backoff *= 2.0
                continue
            response.raise_for_status()
            data = response.json()
            break
        else:
            response.raise_for_status()
        
    embeddings = [emb.get("values", []) for emb in data.get("embeddings", [])]
    return embeddings


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

    t0 = time.time()

    # 1. Embed query via API (uses 0MB of local RAM)
    try:
        embeddings = _call_gemini_embed([query])
        if not embeddings:
            return []
        query_vector = embeddings[0]
    except Exception as e:
        print(f"[RAG ERROR] Gemini embedding call failed: {e}")
        return []

    t1 = time.time()
    print(f"[RAG] Embedding API: {(t1-t0)*1000:.0f}ms")

    # 2. Vector search in Pinecone
    try:
        index = _get_index()
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
    except Exception as e:
        print(f"[RAG ERROR] Pinecone query failed: {e}")
        return []

    matches = results.get("matches", [])
    if not matches:
        return []

    t2 = time.time()
    print(f"[RAG] Pinecone search ({len(matches)} hits): {(t2-t1)*1000:.0f}ms | Total RAG: {(t2-t0)*1000:.0f}ms")

    return [
        {
            "text":     match["metadata"].get("text", ""),
            "score":    float(match.get("score", 0.0)),
            "source":   match["metadata"].get("source", "unknown"),
            "metadata": {k: v for k, v in match["metadata"].items() if k != "text"},
        }
        for match in matches
    ]

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via API. Used by the ingest pipeline."""
    return _call_gemini_embed(texts)
