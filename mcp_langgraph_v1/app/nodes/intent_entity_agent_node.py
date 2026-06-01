from app.graph.state import GraphState
from app.agents.intent_agent import IntentAgent


intent_agent = IntentAgent()


def intent_entity_agent_node(state: GraphState) -> GraphState:
    result = intent_agent.run(
        user_query=state["user_query"],
        normalized_query=state["normalized_query"],
    )

    return {
        **state,
        "intent": result["intent"],
        "intent_confidence": result["intent_confidence"],
        "sub_intent": result.get("sub_intent"),
        "entities": result.get("entities", {}),
        "ambiguities": result.get("ambiguities", []),
        "required_fields": result.get("required_fields", []),
        "trace": state.get("trace", []) + ["intent_entity_complete"],
    }