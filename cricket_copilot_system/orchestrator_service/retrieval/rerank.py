from state import CricketState
from tools.search_retrieval import rerank_candidates


def rq_rerank_node(state: CricketState) -> CricketState:
    retrieval = dict(state.get("retrieval", {}))
    candidates = retrieval.get("candidates", [])

    rr = rerank_candidates(
        user_query=state["message"],
        candidates=candidates
    )

    retrieval["matches"] = rr.get("matches", [])
    retrieval["confidence"] = rr.get("confidence", 0.0)
    retrieval["is_relevant"] = rr.get("is_relevant", False)
    retrieval["rationale"] = rr.get("rationale", "")

    return {"retrieval": retrieval}