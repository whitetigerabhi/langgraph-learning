from state import CricketState
from tools.search_retrieval import rewrite_query


def rq_retry_rewrite_node(state: CricketState) -> CricketState:
    retrieval = dict(state.get("retrieval", {}))

    rewritten = rewrite_query(
        query=f"cricket knowledge answer: {state['message']}",
        memory_summary=state.get("memory_summary", "")
    )

    retrieval["query_rewritten"] = rewritten
    retrieval["attempt"] = 2

    return {"retrieval": retrieval}