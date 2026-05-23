from langgraph.graph import StateGraph, END

from state import CricketState
from nodes.memory_load import memory_load_node
from nodes.supervisor_route import supervisor_route_node
from nodes.analytics_planner import analytics_planner_node
from nodes.call_action_api import call_action_api_node
from nodes.compose_answer import compose_answer_node
from nodes.memory_save import memory_save_node


def build_graph():
    g = StateGraph(CricketState)

    g.add_node("memory_load", memory_load_node)
    g.add_node("supervisor_route", supervisor_route_node)
    g.add_node("analytics_planner", analytics_planner_node)
    g.add_node("call_action_api", call_action_api_node)
    g.add_node("compose_answer", compose_answer_node)
    g.add_node("memory_save", memory_save_node)

    g.set_entry_point("memory_load")
    g.add_edge("memory_load", "supervisor_route")
    g.add_edge("supervisor_route", "analytics_planner")
    g.add_edge("analytics_planner", "call_action_api")
    g.add_edge("call_action_api", "compose_answer")
    g.add_edge("compose_answer", "memory_save")
    g.add_edge("memory_save", END)

    return g.compile()


GRAPH = build_graph()
