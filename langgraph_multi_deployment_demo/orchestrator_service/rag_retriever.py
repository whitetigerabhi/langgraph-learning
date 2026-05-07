import os
import json
import math
from typing import List, Dict, Any
from openai import AzureOpenAI

BASE_DIR = os.path.dirname(__file__)
INDEX_PATH = os.path.join(BASE_DIR, "storage", "rag_index.json")

# Retrieval quality threshold (tune this)
# Typical starting ranges for cosine similarity in small corpora:
# 0.75-0.85 stricter, 0.65-0.75 moderate
RAG_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.72"))

# Debug flag: set RAG_DEBUG=1 to print retrieval diagnostics
RAG_DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"


def _client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


def _embedding_deployment() -> str:
    dep = os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "").strip()
    if not dep:
        raise RuntimeError(
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT is not set.\n"
            "Example:\n"
            "  export AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT='embedding-model'"
        )
    return dep


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _embed_query(query: str) -> List[float]:
    client = _client()
    model = _embedding_deployment()
    resp = client.embeddings.create(model=model, input=[query])
    return resp.data[0].embedding


def retrieve_docs(query: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Returns top_k retrieved chunks PLUS retrieval confidence metrics.
    The agent/runtime can decide whether to USE or IGNORE these results.
    """
    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(f"RAG index not found at {INDEX_PATH}. Run: python3 build_rag_index.py")

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        records = json.load(f).get("records", [])

    qemb = _embed_query(query)

    scored = []
    for rec in records:
        score = _cosine(qemb, rec["embedding"])
        scored.append((score, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max(1, int(top_k))]

    scores = [s for s, _ in top]
    max_score = max(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0

    is_relevant = max_score >= RAG_MIN_SCORE

    if RAG_DEBUG:
        print("\n🔎 RAG DEBUG")
        print(f"Query: {query}")
        print(f"TopK: {top_k} | threshold: {RAG_MIN_SCORE} | max_score: {max_score:.4f} | relevant: {is_relevant}")
        for s, r in top:
            print(f"  {s:.4f} | {r['doc']} | chunk_id={r['chunk_id']} | idx={r['chunk_index']}")

    return {
        "query": query,
        "top_k": int(top_k),
        "threshold_used": RAG_MIN_SCORE,
        "max_score": round(max_score, 4),
        "min_score": round(min_score, 4),
        "avg_score": round(avg_score, 4),
        "is_relevant": bool(is_relevant),
        "matches": [
            {
                "score": round(s, 4),
                "doc": r["doc"],
                "chunk_id": r["chunk_id"],
                "chunk_index": r["chunk_index"],
                "text": r["text"],
            }
            for s, r in top
        ],
    }
