from langgraph.graph import StateGraph, END

from state import CricketState

from nodes.retrieval.rewrite_query import rq_rewrite_node
from nodes.retrieval.hybrid_search import rq_search_node
from nodes.retrieval.rerank import rq_rerank_node
from nodes.retrieval.retry_rewrite import rq_retry_rewrite_node
from nodes.retrieval.package_evidence import rq_package_node
from nodes.retrieval.routing import route_after_rerank


def build_retrieval_graph():
    g = StateGraph(CricketState)

    g.add_node("rq_rewrite", rq_rewrite_node)
    g.add_node("rq_search", rq_search_node)
    g.add_node("rq_rerank", rq_rerank_node)
    g.add_node("rq_retry_rewrite", rq_retry_rewrite_node)
    g.add_node("rq_package", rq_package_node)

    g.set_entry_point("rq_rewrite")
    g.add_edge("rq_rewrite", "rq_search")
    g.add_edge("rq_search", "rq_rerank")

    g.add_conditional_edges(
        "rq_rerank",
        route_after_rerank,
        {
            "retry_rewrite": "rq_retry_rewrite",
            "package": "rq_package",
        },
    )

    g.add_edge("rq_retry_rewrite", "rq_search")
    g.add_edge("rq_package", END)

    return g.compile()


RETRIEVAL_GRAPH = build_retrieval_graph()