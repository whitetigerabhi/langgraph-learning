import os
import json
from typing import Any, Dict, List

from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery


# -----------------------------
# Environment / Clients
# -----------------------------
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]

AOAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AOAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
CHAT_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
EMBED_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]

RETRIEVAL_TOP_N = int(os.environ.get("RETRIEVAL_TOP_N", "8"))
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "4"))
RETRIEVAL_RERANK_MIN = float(os.environ.get("RETRIEVAL_RERANK_MIN", "0.30"))

aoai = AzureOpenAI(
    api_key=AOAI_KEY,
    azure_endpoint=AOAI_ENDPOINT,
    api_version=AOAI_API_VERSION,
)

search = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
    credential=AzureKeyCredential(SEARCH_KEY),
)


# -----------------------------
# Helpers
# -----------------------------
def embed_query(text: str) -> List[float]:
    resp = aoai.embeddings.create(
        model=EMBED_DEPLOYMENT,
        input=[text]
    )
    return resp.data[0].embedding


def rewrite_query(query: str, memory_summary: str = "") -> str:
    """
    Small LLM rewrite to make retrieval more keyword-rich.
    Keep it short, focused, and retrieval-friendly.
    """
    prompt = f"""
Rewrite this cricket question into a short search-friendly query.
If memory summary is useful, use it. Keep the meaning unchanged.

Memory summary:
{memory_summary}

Question:
{query}

Return ONLY the rewritten query.
""".strip()

    resp = aoai.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Return only the rewritten query."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=60,
    )
    return (resp.choices[0].message.content or query).strip()


def hybrid_search(query_text: str, top_n: int = RETRIEVAL_TOP_N) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: keyword + vector.
    Assumes your Search index has fields:
      title, content, entity_type, source_id, chunk_id, embedding
    """
    q_emb = embed_query(query_text)
    vq = VectorizedQuery(vector=q_emb, k_nearest_neighbors=top_n, fields="embedding")

    results = search.search(
        search_text=query_text,
        vector_queries=[vq],
        top=top_n,
        select=["title", "content", "entity_type", "source_id", "chunk_id"],
    )

    candidates = []
    for r in results:
        candidates.append({
            "title": r.get("title"),
            "content": r.get("content"),
            "entity_type": r.get("entity_type"),
            "source_id": r.get("source_id"),
            "chunk_id": r.get("chunk_id"),
            "search_score": float(r.get("@search.score", 0.0)),
        })
    return candidates


def rerank_candidates(user_query: str, candidates: List[Dict[str, Any]], top_k: int = RETRIEVAL_TOP_K) -> Dict[str, Any]:
    """
    LLM reranker:
    returns ranked candidates and a confidence score.
    """
    if not candidates:
        return {"matches": [], "confidence": 0.0, "is_relevant": False, "rationale": "no candidates"}

    packed = []
    for i, c in enumerate(candidates):
        snippet = (c["content"] or "").replace("\n", " ").strip()
        if len(snippet) > 350:
            snippet = snippet[:350] + "..."
        packed.append({
            "i": i,
            "title": c.get("title"),
            "entity_type": c.get("entity_type"),
            "source_id": c.get("source_id"),
            "chunk_id": c.get("chunk_id"),
            "text": snippet,
        })

    prompt = {
        "task": "rerank_cricket_knowledge_snippets",
        "query": user_query,
        "rules": [
            "Score snippet relevance from 0.0 to 1.0",
            "Prefer snippets that directly answer the question",
            "For definitions/explanations, prefer concise rule/glossary content",
            "Return JSON only"
        ],
        "candidates": packed,
        "return_format": {
            "ranked_indices": "best-to-worst candidate indexes",
            "scores": "map index->float",
            "rationale": "short reason"
        }
    }

    resp = aoai.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON."},
            {"role": "user", "content": json.dumps(prompt)},
        ],
        max_tokens=500,
    )

    raw = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        # fallback: use raw search order
        matches = candidates[:top_k]
        return {"matches": matches, "confidence": 0.0, "is_relevant": len(matches) > 0, "rationale": "fallback_search_order"}

    ranked_indices = parsed.get("ranked_indices", [])
    scores = parsed.get("scores", {})
    rationale = parsed.get("rationale", "")

    matches = []
    max_score = 0.0

    for idx in ranked_indices[:top_k]:
        try:
            i = int(idx)
            c = dict(candidates[i])
            rr = float(scores.get(str(i), scores.get(i, 0.0)))
            c["rerank_score"] = rr
            matches.append(c)
            max_score = max(max_score, rr)
        except Exception:
            continue

    is_relevant = max_score >= RETRIEVAL_RERANK_MIN if matches else False

    return {
        "matches": matches,
        "confidence": max_score,
        "is_relevant": is_relevant,
        "rationale": rationale,
    }