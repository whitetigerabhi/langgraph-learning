import os
import json
from typing import Dict, Any, List
from openai import AzureOpenAI

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery


# -----------------------------
# Config
# -----------------------------
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "rag-index-live")

AOAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AOAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")

EMBED_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]
CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")
RERANK_DEPLOYMENT = os.environ.get("AZURE_OPENAI_RERANK_DEPLOYMENT", CHAT_DEPLOYMENT)

RAG_CANDIDATES = int(os.environ.get("RAG_CANDIDATES", "20"))   # retrieve N candidates from Azure Search
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "4"))             # return K snippets after rerank
RAG_ENABLE_REWRITE = os.environ.get("RAG_ENABLE_REWRITE", "1") == "1"
RAG_ENABLE_RERANK = os.environ.get("RAG_ENABLE_RERANK", "1") == "1"

# IMPORTANT: rerank score is 0..1 (LLM-produced), not Azure @search.score
RAG_RERANK_MIN = float(os.environ.get("RAG_RERANK_MIN", "0.30"))

RAG_DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"


# -----------------------------
# Clients
# -----------------------------
aoai = AzureOpenAI(
    api_key=AOAI_KEY,
    azure_endpoint=AOAI_ENDPOINT,
    api_version=AOAI_API_VERSION,
)

search = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_KEY),
)


# -----------------------------
# Helpers
# -----------------------------
def _embed_one(text: str) -> List[float]:
    """Embed a single string using Azure OpenAI embeddings deployment."""
    resp = aoai.embeddings.create(model=EMBED_DEPLOYMENT, input=[text])
    return resp.data[0].embedding


def _rewrite_query(user_query: str) -> str:
    """Use a chat model to rewrite the query for better retrieval."""
    if not RAG_ENABLE_REWRITE:
        return user_query

    prompt = f"""
Rewrite the following question into a search-friendly query for retrieving internal policy/runbook documentation.

Rules:
- Keep it short (<= 18 words).
- Add keywords likely present in internal docs.
- Keep meaning the same.
- Return ONLY the rewritten query.

Question:
{user_query}
""".strip()

    resp = aoai.chat.completions.create(
        model=RERANK_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Return only the rewritten query. No extra text."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=60,
    )
    rq = (resp.choices[0].message.content or "").strip()
    return rq or user_query


def _hybrid_candidates(query_text: str, top_n: int) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval from Azure AI Search:
      - keyword search via search_text
      - vector search via vector_queries (VectorizedQuery)
    """
    q_emb = _embed_one(query_text)
    vq = VectorizedQuery(vector=q_emb, k_nearest_neighbors=int(top_n), fields="embedding")

    results = search.search(
        search_text=query_text,
        vector_queries=[vq],
        top=int(top_n),
        # IMPORTANT: our index schema does NOT have "doc"
        select=["content", "title", "entity_type", "source_table", "source_id", "chunk_index", "chunk_id"],
    )

    out: List[Dict[str, Any]] = []
    for r in results:
        out.append({
            "search_score": float(r.get("@search.score", 0.0)),
            "entity_type": r.get("entity_type"),
            "title": r.get("title"),
            "source_table": r.get("source_table"),
            "source_id": r.get("source_id"),
            "chunk_id": r.get("chunk_id"),
            "chunk_index": r.get("chunk_index"),
            "text": r.get("content"),
        })
    return out


def _llm_rerank(user_query: str, candidates: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    """LLM reranker that scores candidates 0..1 and returns top_k."""
    if (not RAG_ENABLE_RERANK) or (not candidates):
        return {"ranked": candidates[:top_k], "max_score": 0.0, "rationale": "rerank_disabled_or_no_candidates"}

    packed = []
    for i, c in enumerate(candidates):
        snippet = (c.get("text") or "").replace("\n", " ").strip()
        if len(snippet) > 450:
            snippet = snippet[:450] + "..."
        packed.append({
            "i": i,
            "entity_type": c.get("entity_type"),
            "title": c.get("title"),
            "source_table": c.get("source_table"),
            "source_id": c.get("source_id"),
            "chunk_id": c.get("chunk_id"),
            "text": snippet,
        })

    payload = {
        "task": "rerank_snippets",
        "query": user_query,
        "rules": [
            "Score relevance to the query from 0.0 to 1.0",
            "Prefer snippets that directly answer the query (policy/runbook details)",
            "If none answer, score low; do not hallucinate missing details",
            "Return JSON only"
        ],
        "candidates": packed,
        "return_format": {
            "ranked_indices": "list of candidate i in best-to-worst order",
            "scores": "map from i to float score",
            "rationale": "1-2 sentences explaining why the top result is best"
        }
    }

    resp = aoai.chat.completions.create(
        model=RERANK_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON."},
            {"role": "user", "content": json.dumps(payload)},
        ],
        max_tokens=700,
    )

    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
    except Exception:
        return {"ranked": candidates[:top_k], "max_score": 0.0, "rationale": "rerank_json_parse_failed"}

    order = data.get("ranked_indices") or []
    scores = data.get("scores") or {}
    rationale = data.get("rationale") or ""

    ranked: List[Dict[str, Any]] = []
    max_score = 0.0

    if order:
        for idx in order:
            try:
                i = int(idx)
                c = candidates[i]
                s = scores.get(str(i), scores.get(i, 0.0))
                rr = float(s)
            except Exception:
                continue
            max_score = max(max_score, rr)
            c2 = dict(c)
            c2["rerank_score"] = rr
            ranked.append(c2)
    else:
        ranked = candidates[:]
        max_score = 0.0

    return {"ranked": ranked[:top_k], "max_score": max_score, "rationale": rationale}


# -----------------------------
# Public tool entrypoint
# -----------------------------
def retrieve_docs(query: str, top_k: int = None) -> Dict[str, Any]:
    top_k = int(top_k or RAG_TOP_K)

    rewritten = _rewrite_query(query)
    candidates = _hybrid_candidates(rewritten, top_n=max(RAG_CANDIDATES, top_k))

    rr = _llm_rerank(query, candidates, top_k=top_k)
    matches = rr["ranked"]
    max_rerank = float(rr.get("max_score", 0.0))
    is_relevant = (max_rerank >= RAG_RERANK_MIN) if RAG_ENABLE_RERANK else (len(matches) > 0)

    if RAG_DEBUG:
        print("\n🔎 RAG PIPELINE DEBUG")
        print(f"Index: {INDEX_NAME}")
        print(f"Query: {query}")
        print(f"Rewritten: {rewritten}")
        print(f"Candidates: {len(candidates)} | top_k={top_k}")
        print(f"Max rerank: {max_rerank:.3f} | threshold={RAG_RERANK_MIN} | relevant={is_relevant}")
        if rr.get("rationale"):
            print("Rationale:", rr["rationale"])
        for m in matches:
            print(
                f"  rerank={m.get('rerank_score',0):.3f} | search={m.get('search_score',0):.3f} | "
                f"{m.get('entity_type')} | {m.get('title')} | {m.get('source_table')}:{m.get('source_id')} | {m.get('chunk_id')}"
            )

    return {
        "query": query,
        "rewritten_query": rewritten,
        "top_k": top_k,
        "rerank_threshold": RAG_RERANK_MIN,
        "max_rerank_score": round(max_rerank, 4),
        "is_relevant": bool(is_relevant),
        "rationale": rr.get("rationale", ""),
        "matches": [
            {
                "entity_type": m.get("entity_type"),
                "title": m.get("title"),
                "source_table": m.get("source_table"),
                "source_id": m.get("source_id"),
                "chunk_id": m.get("chunk_id"),
                "chunk_index": m.get("chunk_index"),
                "search_score": round(float(m.get("search_score", 0.0)), 4),
                "rerank_score": round(float(m.get("rerank_score", 0.0)), 4),
                "text": m.get("text"),
            }
            for m in matches
        ],
    }
