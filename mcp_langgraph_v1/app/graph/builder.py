from langgraph.graph import StateGraph, START, END

from app.graph.state import GraphState
from app.graph.routing import (
    route_after_router,
    route_after_analytics,
    route_after_retrieval,
)

from app.nodes.intake_node import intake_node
from app.nodes.intent_entity_agent_node import intent_entity_agent_node
from app.nodes.adequacy_check_node import adequacy_check_node
from app.nodes.router_node import router_node
from app.nodes.analytics_agent_node import analytics_agent_node
from app.nodes.retrieval_agent_node import retrieval_agent_node
from app.nodes.fusion_node import fusion_node
from app.nodes.synthesis_node import synthesis_node
from app.nodes.clarify_node import clarify_node


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("intake_node", intake_node)
    graph.add_node("intent_entity_agent_node", intent_entity_agent_node)
    graph.add_node("adequacy_check_node", adequacy_check_node)
    graph.add_node("router_node", router_node)
    graph.add_node("analytics_agent_node", analytics_agent_node)
    graph.add_node("retrieval_agent_node", retrieval_agent_node)
    graph.add_node("fusion_node", fusion_node)
    graph.add_node("synthesis_node", synthesis_node)
    graph.add_node("clarify_node", clarify_node)

    graph.add_edge(START, "intake_node")
    graph.add_edge("intake_node", "intent_entity_agent_node")
    graph.add_edge("intent_entity_agent_node", "adequacy_check_node")
    graph.add_edge("adequacy_check_node", "router_node")

    graph.add_conditional_edges(
        "router_node",
        route_after_router,
        {
            "analytics": "analytics_agent_node",
            "retrieval": "retrieval_agent_node",
            "mixed": "analytics_agent_node",
            "clarify": "clarify_node",
        },
    )

    graph.add_conditional_edges(
        "analytics_agent_node",
        route_after_analytics,
        {
            "to_synthesis": "synthesis_node",
            "to_retrieval": "retrieval_agent_node",
        },
    )

    graph.add_conditional_edges(
        "retrieval_agent_node",
        route_after_retrieval,
        {
            "to_fusion": "fusion_node",
            "to_synthesis": "synthesis_node",
        },
    )

    graph.add_edge("fusion_node", "synthesis_node")
    graph.add_edge("clarify_node", END)
    graph.add_edge("synthesis_node", END)

    return graph.compile()