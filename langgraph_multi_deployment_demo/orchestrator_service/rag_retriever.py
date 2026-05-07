import os
import json
import math
from typing import List, Dict, Any
from openai import AzureOpenAI

BASE_DIR = os.path.dirname(__file__)
INDEX_PATH = os.path.join(BASE_DIR, "storage", "rag_index.json")


# -----------------------------
# CLIENT
# -----------------------------
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
            "Run:\n"
            "export AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=\"embedding-model\""
        )
    return dep


# -----------------------------
# COSINE SIMILARITY
# -----------------------------
def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# -----------------------------
# EMBED QUERY
# -----------------------------
def _embed_query(query: str) -> List[float]:
    client = _client()
    model = _embedding_deployment()

    resp = client.embeddings.create(
        model=model,
        input=[query]
    )

    return resp.data[0].embedding


# -----------------------------
# RETRIEVE DOCUMENTS
# -----------------------------
def retrieve_docs(query: str, top_k: int = 4) -> Dict[str, Any]:

    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(
            f"RAG index not found at {INDEX_PATH}. "
            "Run: python3 build_rag_index.py"
        )

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])

    # Embed user query
    query_embedding = _embed_query(query)

    # Score all chunks
    scored = []
    for rec in records:
        score = _cosine(query_embedding, rec["embedding"])
        scored.append((score, rec))

    # Sort by similarity
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top K
    top = scored[:max(1, int(top_k))]

    return {
        "query": query,
        "top_k": top_k,
        "matches": [
            {
                "score": round(score, 4),
                "doc": rec["doc"],
                "chunk_id": rec["chunk_id"],
                "chunk_index": rec["chunk_index"],
                "text": rec["text"]
            }
            for score, rec in top
        ]
    }