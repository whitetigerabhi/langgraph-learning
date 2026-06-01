from app.graph.state import GraphState
from app.agents.clarify_agent import ClarifyAgent


clarify_agent = ClarifyAgent()


def clarify_node(state: GraphState) -> GraphState:
    question = clarify_agent.run(
        user_query=state["user_query"],
        missing_fields=state.get("missing_fields", []),
        ambiguities=state.get("ambiguities", []),
    )

    return {
        **state,
        "clarification_question": question,
        "final_answer": question,
        "trace": state.get("trace", []) + ["clarify_complete"],
    }