from state import CricketState


def rq_package_node(state: CricketState) -> CricketState:
    retrieval = dict(state.get("retrieval", {}))
    matches = retrieval.get("matches", [])

    citations = []
    for m in matches:
        citations.append({
            "source": "azure_ai_search",
            "source_id": m.get("source_id"),
            "chunk_id": m.get("chunk_id"),
            "title": m.get("title"),
        })

    retrieval["citations"] = citations

    return {"retrieval": retrieval}