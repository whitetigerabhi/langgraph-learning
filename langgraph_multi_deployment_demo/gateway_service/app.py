from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from uuid import uuid4

from policy import preflight_rbac
from content_safety import should_block
from intent_classifier import classify_intent
from orchestrator_client import run_orchestrator

app = FastAPI(title="Gateway Service (Guardrails + Intent + Routing)", version="1.0")

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

    # 1) RBAC / destructive-policy preflight
    ok, reason = preflight_rbac(role, msg)
    if not ok:
        raise HTTPException(status_code=403, detail=reason)

    # 2) Azure AI Content Safety input guardrail
    block_in, cs_in = should_block(msg)
    if block_in:
        raise HTTPException(status_code=403, detail=f"BLOCK: Unsafe input detected. Please try again. details={cs_in}")

    # 3) Cheap LLM intent classification
    intent_data = classify_intent(msg)
    intent = intent_data.get("intent", "invalid")

    if intent == "invalid":
        return {
            "thread_id": req.thread_id or "",
            "intent": "invalid",
            "answer": "I can help with weather, cricket/IPL scores, or general questions. Please rephrase.",
            "meta": {"content_safety_input": cs_in, "intent_data": intent_data},
        }

    # 4) Ensure thread_id (backend generated if not provided)
    thread_id = req.thread_id or str(uuid4())

    # 5) Call orchestrator
    hints = {"intent": intent, "location": intent_data.get("location", ""), "team": intent_data.get("team", "")}
    out = run_orchestrator(thread_id=thread_id, query=msg, user_role=role, hints=hints)

    ans = out.get("answer", "") or ""

    # 6) Optional: Azure AI Content Safety output guardrail at gateway too
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
``