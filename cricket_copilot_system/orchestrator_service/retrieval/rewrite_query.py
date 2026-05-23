from state import CricketState
from tools.search_retrieval import rewrite_query


def rq_rewrite_node(state: CricketState) -> CricketState:
    retrieval = dict(state.get("retrieval", {}))

    rewritten = rewrite_query(
        query=state["message"],
        memory_summary=state.get("memory_summary", "")
    )

    retrieval["query_rewritten"] = rewritten
    retrieval["attempt"] = 1

    return {"retrieval": retrieval}