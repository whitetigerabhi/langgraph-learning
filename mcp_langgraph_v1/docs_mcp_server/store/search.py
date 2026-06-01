import re
from docs_mcp_server.store.ingest import load_docs


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def search_docs_store(query: str, top_k: int = 5, filters: dict | None = None) -> dict:
    filters = filters or {}
    docs = load_docs()
    query_tokens = tokenize(query)

    matches = []

    for doc in docs:
        if "doc_type" in filters and doc["doc_type"] != filters["doc_type"]:
            continue

        for chunk in doc["chunks"]:
            chunk_tokens = tokenize(chunk["text"])
            overlap = len(query_tokens.intersection(chunk_tokens))
            if overlap > 0:
                score = overlap / max(len(query_tokens), 1)
                matches.append(
                    {
                        "doc_id": doc["doc_id"],
                        "doc_type": doc["doc_type"],
                        "title": doc["title"],
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "score": round(score, 4),
                    }
                )

    matches = sorted(matches, key=lambda x: x["score"], reverse=True)[:top_k]
    return {"matches": matches}