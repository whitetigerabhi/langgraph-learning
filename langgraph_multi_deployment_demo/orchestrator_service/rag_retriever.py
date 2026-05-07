import os
import json
import math
from typing import List, Dict, Any
from openai import AzureOpenAI

BASE_DIR = os.path.dirname(__file__)
INDEX_PATH = os.path.join(BASE_DIR, "storage", "rag_index.json")

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
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT is not set. "
            "Set it to your embeddings deployment before using RAG."
        )
    return dep

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)

def _embed_query(query: str) -> Listclient = _client()
    model = _embedding_deployment()
    resp = client.embeddings.create(model=model, input=[query])
    return resp.data[0].embedding

def retrieve_docs(query: str, top_k: int = 4) -> Dict[str, Any]:
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

    return {
        "query": query,
        "top_k": int(top_k),
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