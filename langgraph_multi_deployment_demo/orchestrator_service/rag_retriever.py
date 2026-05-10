import os
from typing import Dict, Any, List
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery  # modern SDK vector query [1](https://stackoverflow.com/questions/76419780/azure-cognitive-search-in-python-using-vector-embeddings-error)

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "rag-index")

AOAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AOAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
EMBED_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]

RAG_DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"

# NOTE: Azure Search @search.score is not cosine similarity. Treat this as tunable.
RAG_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.65"))

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

def retrieve_docs(query: str, top_k: int = 4) -> Dict[str, Any]:
    # 1) Embed query
    q_emb = aoai.embeddings.create(model=EMBED_DEPLOYMENT, input=[query]).data[0].embedding

    # 2) Hybrid query: keyword + vector
    vq = VectorizedQuery(vector=q_emb, k_nearest_neighbors=int(top_k), fields="embedding")

    results = search.search(
        search_text=query,
        vector_queries=[vq],
        top=int(top_k),
        select=["content", "doc", "chunk_index", "chunk_id"],
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
        "max_score": round(max_score, 4),
        "threshold_used": RAG_MIN_SCORE,
        "is_relevant": bool(is_relevant),
        "matches": matches,
    }
