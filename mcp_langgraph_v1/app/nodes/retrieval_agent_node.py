from app.graph.state import GraphState
from app.agents.retrieval_agent import RetrievalAgent
from app.mcp_clients.docs_client import DocsMCPClient


retrieval_agent = RetrievalAgent()
docs_client = DocsMCPClient()


async def retrieval_agent_node(state: GraphState) -> GraphState:
    plan = retrieval_agent.plan(
        user_query=state["user_query"],
        entities=state.get("entities", {}),
    )

    result = await docs_client.search_docs(
        query=plan["search_query"],
        top_k=plan.get("top_k", 5),
        filters=plan.get("filters", {}),
    )

    evidence_items = [
        {
            "source_type": "document",
            "source_id": f'{m["doc_id"]}:{m["chunk_id"]}',
            "content": m,
            "relevance_score": m.get("score"),
        }
        for m in result.get("matches", [])
    ]

    return {
        **state,
        "retrieval_result": result,
        "evidence": state.get("evidence", []) + evidence_items,
        "trace": state.get("trace", []) + ["retrieval_complete"],
    }