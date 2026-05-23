from langgraph.graph import StateGraph, END

from state import CricketState

from nodes.memory_load import memory_load_node
from nodes.supervisor_route import supervisor_route_node
from nodes.analytics_planner import analytics_planner_node
from nodes.call_action_api import call_action_api_node
from nodes.compose_answer import compose_answer_node
from nodes.memory_save import memory_save_node

from subgraphs.retrieval_graph import RETRIEVAL_GRAPH


def route_after_supervisor(state: CricketState):
    route = state.get("route", "trivia")
    if route == "analytics":
        return "analytics"
    return "trivia"


def build_graph():
    g = StateGraph(CricketState)

    g.add_node("memory_load", memory_load_node)
    g.add_node("supervisor_route", supervisor_route_node)

    # Analytics path
    g.add_node("analytics_planner", analytics_planner_node)
    g.add_node("call_action_api", call_action_api_node)

    # Retrieval path (subgraph as node)
    g.add_node("retrieval_subgraph", RETRIEVAL_GRAPH)

    # Shared tail
    g.add_node("compose_answer", compose_answer_node)
    g.add_node("memory_save", memory_save_node)

    g.set_entry_point("memory_load")
    g.add_edge("memory_load", "supervisor_route")

    g.add_conditional_edges(
        "supervisor_route",
        route_after_supervisor,
        {
            "analytics": "analytics_planner",
            "trivia": "retrieval_subgraph",
        },
    )

    g.add_edge("analytics_planner", "call_action_api")
    g.add_edge("call_action_api", "compose_answer")

    g.add_edge("retrieval_subgraph", "compose_answer")

    g.add_edge("compose_answer", "memory_save")
    g.add_edge("memory_save", END)

    return g.compile()


GRAPH = build_graph()