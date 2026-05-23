from state import CricketState
from tools.search_retrieval import hybrid_search


def rq_search_node(state: CricketState) -> CricketState:
    retrieval = dict(state.get("retrieval", {}))
    query_text = retrieval.get("query_rewritten", state["message"])

    candidates = hybrid_search(query_text=query_text)

    retrieval["candidates"] = candidates
    retrieval["candidate_count"] = len(candidates)

    return {"retrieval": retrieval}