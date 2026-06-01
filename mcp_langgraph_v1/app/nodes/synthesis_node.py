from app.graph.state import GraphState
from app.agents.synthesis_agent import SynthesisAgent


synthesis_agent = SynthesisAgent()


def synthesis_node(state: GraphState) -> GraphState:
    final_answer = synthesis_agent.run(
        user_query=state["user_query"],
        route=state.get("route", "clarify"),
        evidence=state.get("evidence", []),
        analytics_result=state.get("analytics_result"),
        retrieval_result=state.get("retrieval_result"),
    )

    return {
        **state,
        "final_answer": final_answer,
        "trace": state.get("trace", []) + ["synthesis_complete"],
    }