from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, Any, Dict
from uuid import uuid4
import re

from graph import GRAPH

app = FastAPI(title="Plan B1 Agent + Deterministic Gateway + Time Travel", version="1.0")


class RunRequest(BaseModel):
    thread_id: str
    query: str

class ContinueRequest(BaseModel):
    thread_id: str
    user_input: str

class ApproveRequest(BaseModel):
    thread_id: str
    approval_token: str

class ReplayRequest(BaseModel):
    checkpoint_id: str

class ForkRequest(BaseModel):
    checkpoint_id: Optional[str] = None
    new_thread_id: Optional[str] = None

class PatchRequest(BaseModel):
    checkpoint_id: Optional[str] = None
    update: Dict[str, Any]


def cfg(thread_id: str, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
    c = {"configurable": {"thread_id": thread_id}}
    if checkpoint_id:
        c["configurable"]["checkpoint_id"] = checkpoint_id
    return c


DESTRUCTIVE_PATTERNS = [
    r"\bdrop\s+table\b",
    r"\btruncate\s+table\b",
    r"\bdelete\s+table\b",
    r"\bdestroy\s+repo\b",
    r"\brm\s+-rf\b",
    r"\bwipe\b",
]

UNSAFE_PATTERNS = [
    r"\bmake\s+a\s+bomb\b",
    r"\bbuild\s+a\s+bomb\b",
]

def _matches_any(patterns, text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in patterns)


def preflight(role: str, query: str):
    """
    Deterministic gateway:
    - BLOCK unsafe content for everyone.
    - BLOCK destructive actions unless admin.
    - Ask user to try again (your requirement).
    """
    role = (role or "user").lower()

    if _matches_any(UNSAFE_PATTERNS, query):
        raise HTTPException(
            status_code=403,
            detail="BLOCK: I can’t help with that. Please try again with a safe request."
        )

    if _matches_any(DESTRUCTIVE_PATTERNS, query) and role != "admin":
        raise HTTPException(
            status_code=403,
            detail="BLOCK: Destructive actions are restricted to admins. Please try again with a non-destructive request."
        )


def _find_snapshot_by_checkpoint(thread_id: str, checkpoint_id: str, limit: int = 500):
    for s in GRAPH.get_state_history(cfg(thread_id), limit=limit):
        if s.config["configurable"].get("checkpoint_id") == checkpoint_id:
            return s
    return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/threads/new")
def new_thread():
    return {"thread_id": str(uuid4())}


@app.post("/run")
def run(req: RunRequest, x_user_role: Optional[str] = Header(default="user")):
    # 1) Front-door deterministic block
    preflight(x_user_role, req.query)

    # 2) Run LangGraph workflow
    result = GRAPH.invoke(
        {"query": req.query, "user_role": (x_user_role or "user")},
        config=cfg(req.thread_id),
    )

    return {
        "thread_id": req.thread_id,
        "role": x_user_role,
        "blocked": result.get("blocked", False),
        "waiting_for_user": result.get("waiting_for_user", False),
        "needs_approval": bool(result.get("needs_approval", False) and not result.get("approved", False)),
        "approval_token": result.get("approval_token", ""),
        "answer": result.get("answer", ""),
        "state": result,
    }


@app.post("/continue")
def continue_after_clarify(req: ContinueRequest):
    """
    Continue after ASK_CLARIFICATION:
    - patch clarified value into state
    - clear waiting_for_user
    - advance step_index by 1 (move past clarification step)
    - resume the graph
    """
    snap = GRAPH.get_state(cfg(req.thread_id))
    values = snap.values or {}

    if not values.get("waiting_for_user"):
        raise HTTPException(status_code=400, detail="Thread is not waiting for user input.")

    key = (values.get("clarify_key") or "other").strip().lower()
    clarified = dict(values.get("clarified") or {})
    clarified[key] = req.user_input

    if not hasattr(GRAPH, "update_state"):
        raise HTTPException(status_code=500, detail="GRAPH.update_state not available; upgrade langgraph")

    # advance step index past the clarification step
    GRAPH.update_state(cfg(req.thread_id), {
        "clarified": clarified,
        "waiting_for_user": False,
        "clarification_question": "",
        "pending_step_type": "",
        "step_index": int(values.get("step_index", 0)) + 1,
    })

    # resume
    result = GRAPH.invoke({"query": values.get("query", ""), "user_role": values.get("user_role", "user")}, config=cfg(req.thread_id))
    return {"thread_id": req.thread_id, "answer": result.get("answer", ""), "state": result}


@app.post("/approve")
def approve(req: ApproveRequest):
    """
    Approve after REQUIRE_APPROVAL:
    - validate token
    - set approved=True
    - advance step_index by 1
    - resume
    """
    snap = GRAPH.get_state(cfg(req.thread_id))
    values = snap.values or {}

    if not values.get("needs_approval") or values.get("approved"):
        raise HTTPException(status_code=400, detail="Thread does not need approval.")

    if values.get("approval_token") != req.approval_token:
        raise HTTPException(status_code=400, detail="Invalid approval_token.")

    if not hasattr(GRAPH, "update_state"):
        raise HTTPException(status_code=500, detail="GRAPH.update_state not available; upgrade langgraph")

    GRAPH.update_state(cfg(req.thread_id), {
        "approved": True,
        "pending_step_type": "",
        "step_index": int(values.get("step_index", 0)) + 1,
    })

    result = GRAPH.invoke({"query": values.get("query", ""), "user_role": values.get("user_role", "user")}, config=cfg(req.thread_id))
    return {"thread_id": req.thread_id, "approved": True, "answer": result.get("answer", ""), "state": result}


# -------- Time travel endpoints --------

@app.get("/threads/{thread_id}/state")
def get_state(thread_id: str):
    snap = GRAPH.get_state(cfg(thread_id))
    return {
        "thread_id": thread_id,
        "checkpoint_id": snap.config["configurable"].get("checkpoint_id"),
        "values": snap.values,
        "next": list(snap.next),
    }


@app.get("/threads/{thread_id}/history")
def get_history(thread_id: str, limit: int = 20):
    snaps = []
    for s in GRAPH.get_state_history(cfg(thread_id), limit=limit):
        snaps.append({
            "checkpoint_id": s.config["configurable"].get("checkpoint_id"),
            "next": list(s.next),
            "created_at": getattr(s, "created_at", None),
            "metadata": getattr(s, "metadata", None),
            "values": s.values,
        })
    return {"thread_id": thread_id, "count": len(snaps), "snapshots": snaps}


@app.post("/threads/{thread_id}/replay")
def replay(thread_id: str, req: ReplayRequest):
    snap = _find_snapshot_by_checkpoint(thread_id, req.checkpoint_id)
    if not snap:
        raise HTTPException(status_code=404, detail="checkpoint_id not found")

    # replay from checkpoint config (nodes after this may re-run)
    result = GRAPH.invoke(None, snap.config)
    return {"thread_id": thread_id, "replayed_from": req.checkpoint_id, "answer": result.get("answer", ""), "state": result}


@app.post("/threads/{thread_id}/fork")
def fork(thread_id: str, req: ForkRequest):
    new_thread_id = req.new_thread_id or str(uuid4())

    if req.checkpoint_id:
        snap = _find_snapshot_by_checkpoint(thread_id, req.checkpoint_id)
        if not snap:
            raise HTTPException(status_code=404, detail="checkpoint_id not found")
        base_values = snap.values
        base_ckpt = req.checkpoint_id
    else:
        latest = GRAPH.get_state(cfg(thread_id))
        base_values = latest.values
        base_ckpt = latest.config["configurable"].get("checkpoint_id")

    if not hasattr(GRAPH, "update_state"):
        raise HTTPException(status_code=500, detail="GRAPH.update_state not available; upgrade langgraph")

    GRAPH.update_state(cfg(new_thread_id), base_values)
    return {"source_thread_id": thread_id, "forked_thread_id": new_thread_id, "forked_from_checkpoint_id": base_ckpt}


@app.post("/threads/{thread_id}/patch")
def patch(thread_id: str, req: PatchRequest):
    if not hasattr(GRAPH, "update_state"):
        raise HTTPException(status_code=500, detail="GRAPH.update_state not available; upgrade langgraph")

    target = cfg(thread_id, req.checkpoint_id) if req.checkpoint_id else cfg(thread_id)
    GRAPH.update_state(target, req.update)

    snap = GRAPH.get_state(cfg(thread_id))
    return {"thread_id": thread_id, "new_checkpoint_id": snap.config["configurable"].get("checkpoint_id"), "values": snap.values}