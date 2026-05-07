cd ~/langgraph-learning/langgraph_multi_deployment_demo/gateway_service

cat > app.py <<'PY'
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from uuid import uuid4

from policy import preflight_rbac
from content_safety import should_block
from intent_classifier import classify_intent
from orchestrator_client import run_orchestrator

app = FastAPI(title="Gateway Service (Content Safety + Intent + Routing)", version="1.0")


class ChatRequest(BaseModel):
    thread_id: str | None = None
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest, x_user_role: str = Header(default="user")):
    role = (x_user_role or "user").lower()
    msg = (req.message or "").strip()

    # 1) RBAC / destructive gate
    ok, reason = preflight_rbac(role, msg)
    if not ok:
        raise HTTPException(status_code=403, detail=reason)

    # 2) Content Safety input guardrail (hard gate)
    block_in, cs_in = should_block(msg)
    if block_in:
        raise HTTPException(status_code=403, detail=f"BLOCK: Unsafe input detected. Please try again. details={cs_in}")

    # 3) Intent classification (SOFT signal only)
    intent_data = classify_intent(msg)
    intent = (intent_data.get("intent") or "invalid").strip().lower()

    # IMPORTANT: Do NOT block requests just because classifier says "invalid".
    # Treat it as "general" and let orchestrator decide (agentic behavior + RAG).
    if intent == "invalid":
        intent = "general"

    thread_id = req.thread_id or str(uuid4())

    hints = {
        "intent": intent,
        "location": intent_data.get("location", ""),
        "team": intent_data.get("team", ""),
        "classifier_source": intent_data.get("source", ""),
        "classifier_confidence": intent_data.get("confidence", 0.0),
    }

    # 4) Call orchestrator
    out = run_orchestrator(thread_id=thread_id, query=msg, user_role=role, hints=hints)
    ans = out.get("answer", "") or ""

    # 5) Optional output guardrail at gateway too
    block_out, cs_out = should_block(ans) if ans else (False, {})
    if block_out:
        raise HTTPException(status_code=403, detail=f"BLOCK: Output flagged. Please try again. details={cs_out}")

    return {
        "thread_id": thread_id,
        "intent": intent,
        "answer": ans,
        "meta": {
            "content_safety_input": cs_in,
            "content_safety_output": cs_out,
            "intent_data": intent_data,
            "orchestrator_meta": out.get("meta", {}),
        },
    }
PY