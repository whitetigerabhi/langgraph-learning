from fastapi import FastAPI

from app.schemas.api import ChatRequest, ChatResponse
from app.graph.builder import build_graph


app = FastAPI(title="MCP LangGraph v1")
graph = build_graph()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    initial_state = {
        "thread_id": req.thread_id,
        "user_query": req.message,
        "trace": [],
        "warnings": [],
        "errors": [],
        "evidence": [],
        "planned_tool_calls": [],
        "executed_tool_calls": [],
    }

    result = await graph.ainvoke(initial_state)

    return ChatResponse(
        thread_id=req.thread_id,
        final_answer=result.get("final_answer", ""),
        trace=result.get("trace", []),
        route=result.get("route"),
    )