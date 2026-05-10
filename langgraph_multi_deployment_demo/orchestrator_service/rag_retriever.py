import os
from typing import Dict, Any, List
from openai import AzureOpenAI

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# ---------------- Config ----------------
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "rag-index")

AOAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AOAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
EMBED_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]

RAG_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.65"))
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

# ---------------- Vector query helper ----------------
def _make_vector_query(embedding: List[float], top_k: int):
    """
    Azure Search SDK version differences:
    - Newer versions prefer VectorizedQuery objects.
    - Older builds sometimes accept dicts.

    We try VectorizedQuery first; fallback to dict if import isn't available.
    """
    try:
        from azure.search.documents.models import VectorizedQuery
        return [VectorizedQuery(vector=embedding, k_nearest_neighbors=top_k, fields="embedding")]
    except Exception:
        # Fallback form (works in some SDK versions)
        return [{"vector": embedding, "k": top_k, "fields": "embedding"}]

# ---------------- Public API ----------------
def retrieve_docs(query: str, top_k: int = 4) -> Dict[str, Any]:
    # 1) Embed the query
    q_emb = aoai.embeddings.create(model=EMBED_DEPLOYMENT, input=[query]).data[0].embedding

    # 2) HYBRID retrieval:
    #    - search_text uses classic keyword/BM25
    #    - vector_queries uses embedding similarity
    vector_queries = _make_vector_query(q_emb, int(top_k))

    results = search.search(
        search_text=query,                # keyword component
        vector_queries=vector_queries,    # vector component
        select=["content", "doc", "chunk_index", "chunk_id"],
        top=int(top_k),
    )

    matches = []
    scores = []

    for r in results:
        score = float(r.get("@search.score", 0.0))
        scores.append(score)

        matches.append({
            "score": round(score, 4),
            "doc": r.get("doc"),
            "chunk_id": r.get("chunk_id"),
            "chunk_index": r.get("chunk_index"),
            "text": r.get("content"),
        })

    max_score = max(scores) if scores else 0.0
    is_relevant = max_score >= RAG_MIN_SCORE

    if RAG_DEBUG:
        print("\n🔎 HYBRID RETRIEVAL DEBUG")
        print(f"Query: {query}")
        print(f"top_k={top_k} | max_score={max_score:.4f} | threshold={RAG_MIN_SCORE} | relevant={is_relevant}")
        for m in matches:
            print(f"  {m['score']:.4f} | {m['doc']} | {m['chunk_id']} | idx={m['chunk_index']}")

    return {
        "query": query,
        "top_k": int(top_k),
        "threshold_used": RAG_MIN_SCORE,
        "max_score": round(max_score, 4),
        "is_relevant": bool(is_relevant),
        "matches": matches,
    }