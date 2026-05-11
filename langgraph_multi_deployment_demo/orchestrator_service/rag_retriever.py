import os
import json
from typing import Dict, Any, List
from openai import AzureOpenAI

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery  # modern vector query class [2](https://stackoverflow.com/questions/76419780/azure-cognitive-search-in-python-using-vector-embeddings-error)

# ---------------- Config ----------------
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "rag-index")

AOAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AOAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
EMBED_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]

CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")
RERANK_DEPLOYMENT = os.environ.get("AZURE_OPENAI_RERANK_DEPLOYMENT", CHAT_DEPLOYMENT)

RAG_CANDIDATES = int(os.environ.get("RAG_CANDIDATES", "20"))
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "4"))
RAG_ENABLE_REWRITE = os.environ.get("RAG_ENABLE_REWRITE", "1") == "1"
RAG_ENABLE_RERANK = os.environ.get("RAG_ENABLE_RERANK", "1") == "1"

# This threshold is for LLM rerank score (0..1), not Azure Search score
RAG_RERANK_MIN = float(os.environ.get("RAG_RERANK_MIN", "0.55"))

RAG_DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"


# ---------------- Clients ----------------
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


# ---------------- Helpers ----------------
def _embed(text: str) -> Listreturn aoai.embeddings.create(model=EMBED_DEPLOYMENT, input=[text]).data[0].embedding


def _rewrite_query(user_query: str) -> str:
    """
    Query rewriting for retrieval:
    - adds keywords likely present in docs
    - removes fluff
    - keeps it short
    """
    if not RAG_ENABLE_REWRITE:
        return user_query

    prompt = f"""
Rewrite the following question into a search-friendly query for retrieving internal policy/runbook/product documentation.
Rules:
- Keep it short (<= 18 words).
- Add missing keywords that would appear in internal docs.
- Keep meaning the same.
- Return ONLY the rewritten query.

Question:
{user_query}
""".strip()

    resp = aoai.chat.completions.create(
        model=RERANK_DEPLOYMENT,
        messages=[{"role": "system", "content": "Return only the rewritten query."},
                  {"role": "user", "content": prompt}],
        max_tokens=60,
    )
    rq = (resp.choices[0].message.content or "").strip()
    return rq or user_query


def _hybrid_search(query_text: str, top_n: int) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval:
    - search_text=query_text  (keyword/BM25)
    - vector_queries=[VectorizedQuery(...)] (vector similarity)
    """
    q_emb = _embed(query_text)
    vq = VectorizedQuery(vector=q_emb, k_nearest_neighbors=top_n, fields="embedding")

    results = search.search(
        search_text=query_text,
        vector_queries=[vq],
        top=top_n,
        select=["content", "doc", "chunk_index", "chunk_id"],
    )

    candidates = []
    for r in results:
        candidates.append({
            "search_score": float(r.get("@search.score", 0.0)),
            "doc": r.get("doc"),
            "chunk_id": r.get("chunk_id"),
            "chunk_index": r.get("chunk_index"),
            "text": r.get("content"),
        })
    return candidates


def _llm_rerank(user_query: str, candidates: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    """
    Rerank using LLM as a cross-encoder:
    - Provide (query + snippets)
    - Ask for relevance score per snippet and an ordered top_k list
    Returns:
      { "ranked": [...], "max_score": float }
    """
    if not RAG_ENABLE_RERANK or not candidates:
        return {"ranked": candidates[:top_k], "max_score": 0.0}

    # Keep prompt tight: provide only ids + short text
    items = []
    for i, c in enumerate(candidates):
        snippet = (c["text"] or "").replace("\n", " ").strip()
        if len(snippet) > 450:
            snippet = snippet[:450] + "..."
        items.append({
            "i": i,
            "doc": c.get("doc"),
            "chunk_id": c.get("chunk_id"),
            "text": snippet
        })

    prompt = {
        "task": "rerank_snippets",
        "rules": [
            "Score relevance to the query from 0.0 to 1.0",
            "Prefer snippets that directly contain policy/runbook details answering the query",
            "Do not hallucinate missing policy details",
            "Return JSON only"
        ],
        "query": user_query,
        "candidates": items,
        "return_format": {
            "ranked_indices": "list of candidate i in best-to-worst order",
            "scores": "map from i to float score",
            "rationale": "1-2 sentences explaining top choice"
        }
    }

    resp = aoai.chat.completions.create(
        model=RERANK_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON."},
            {"role": "user", "content": json.dumps(prompt)},
        ],
        max_tokens=600,
    )

    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
    except Exception:
        # If model returns non-JSON, fallback to original ordering
        return {"ranked": candidates[:top_k], "max_score": 0.0, "rationale": "rerank_failed_fallback"}

    order = data.get("ranked_indices") or []
    scores_map = data.get("scores") or {}
    rationale = data.get("rationale") or ""

    ranked = []
    max_score = 0.0

    # Build ranked list using order; fallback if order empty
    if order:
        for idx in order:
            try:
                i = int(idx)
                c = candidates[i]
                rr = float(scores_map.get(str(i), scores_map.get(i, 0.0)))
            except Exception:
                continue
            max_score = max(max_score, rr)
            out = dict(c)
            out["rerank_score"] = rr
            ranked.append(out)
    else:
        ranked = candidates[:]
        max_score = 0.0

    return {"ranked": ranked[:top_k], "max_score": max_score, "rationale": rationale}


# ---------------- Public API ----------------
def retrieve_docs(query: str, top_k: int = None) -> Dict[str, Any]:
    """
    Tool callable by the orchestrator agent.
    Returns top-k chunks, plus quality signals for gating.
    """
    top_k = int(top_k or RAG_TOP_K)

    rewritten = _rewrite_query(query)
    candidates = _hybrid_search(rewritten, top_n=max(top_k, RAG_CANDIDATES))

    reranked = _llm_rerank(query, candidates, top_k=top_k)
    matches = reranked["ranked"]
    max_rerank = float(reranked.get("max_score", 0.0))

    is_relevant = max_rerank >= RAG_RERANK_MIN if RAG_ENABLE_RERANK else (len(matches) > 0)

    if RAG_DEBUG:
        print("\n🔎 RAG PIPELINE DEBUG")
        print(f"Query: {query}")
        print(f"Rewritten: {rewritten}")
        print(f"Candidates: {len(candidates)} | top_k={top_k}")
        print(f"Max rerank: {max_rerank:.3f} | threshold={RAG_RERANK_MIN} | relevant={is_relevant}")
        if reranked.get("rationale"):
            print("Rationale:", reranked["rationale"])
        for m in matches:
            print(f"  rerank={m.get('rerank_score',0):.3f} | search={m.get('search_score',0):.3f} | {m.get('doc')} | {m.get('chunk_id')}")

    return {
        "query": query,
        "rewritten_query": rewritten,
        "top_k": top_k,
        "rerank_threshold": RAG_RERANK_MIN,
        "max_rerank_score": round(max_rerank, 4),
        "is_relevant": bool(is_relevant),
        "rationale": reranked.get("rationale", ""),
        "matches": [
            {
                "doc": m.get("doc"),
                "chunk_id": m.get("chunk_id"),
                "chunk_index": m.get("chunk_index"),
                "search_score": round(float(m.get("search_score", 0.0)), 4),
                "rerank_score": round(float(m.get("rerank_score", 0.0)), 4),
                "text": m.get("text"),
            }
            for m in matches
        ],
    }