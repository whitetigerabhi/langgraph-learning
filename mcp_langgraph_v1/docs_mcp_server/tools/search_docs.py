import json
from docs_mcp_server.store.search import search_docs_store


def search_docs(query: str, top_k: int, filters_json: str) -> str:
    filters = json.loads(filters_json) if filters_json else {}
    result = search_docs_store(query=query, top_k=top_k, filters=filters)
    return json.dumps(result)