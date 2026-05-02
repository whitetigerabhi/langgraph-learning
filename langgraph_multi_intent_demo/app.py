from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict, List
from uuid import uuid4

from graph import GRAPH

app = FastAPI(title="LangGraph Multi-Intent Demo (Time Travel)", version="0.3")


# -----------------------------
# Request models
# -----------------------------
class RunRequest(BaseModel):
    thread_id: str
    query: str


class ReplayRequest(BaseModel):
    checkpoint_id: str


class ForkRequest(BaseModel):
    # Optional: fork from a specific checkpoint_id; if omitted forks from latest
    checkpoint_id: Optional[str] = None
    # Optional: caller can specify new thread_id, else generated
    new_thread_id: Optional[str] = None


class PatchRequest(BaseModel):
    # Optional: patch from a specific checkpoint (creates a new branch checkpoint)
    checkpoint_id: Optional[str] = None
    # Partial state updates to apply
    update: Dict[str, Any]
    # Optional: which node to treat as the "writer" (advanced; safe to omit)
    as_node: Optional[str] = None


# -----------------------------
# Helpers
# -----------------------------
def cfg(thread_id: str, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
    c = {"configurable": {"thread_id": thread_id}}
    if checkpoint_id:
        c["configurable"]["checkpoint_id"] = checkpoint_id
    return c


def _find_snapshot_by_checkpoint(thread_id: str, checkpoint_id: str, limit: int = 500):
    """Search history for a snapshot with matching checkpoint_id."""
    for s in GRAPH.get_state_history(cfg(thread_id), limit=limit):
        if s.config["configurable"].get("checkpoint_id") == checkpoint_id:
            return s
    return None


# -----------------------------
# Baseline endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run(req: RunRequest):
    """
    Normal execution: runs graph on a thread, persists checkpoints automatically.
    """
    result = GRAPH.invoke({"query": req.query}, config=cfg(req.thread_id))
    return {"thread_id": req.thread_id, "query": req.query, "answer": result.get("answer", ""), "state": result}


@app.get("/threads/new")
def new_thread():
    """
    Backend-generated thread_id (best practice).
    """
    return {"thread_id": str(uuid4())}


@app.get("/threads/{thread_id}/state")
def get_state(thread_id: str):
    """
    Get current state snapshot for a thread.
    """
    snap = GRAPH.get_state(cfg(thread_id))
    return {
        "thread_id": thread_id,
        "values": snap.values,
        "next": list(snap.next),
        "checkpoint_id": snap.config["configurable"].get("checkpoint_id"),
    }


@app.get("/threads/{thread_id}/history")
def get_history(thread_id: str, limit: int = 20):
    """
    Get checkpoint history for a thread (newest-first).
    """
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")

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


# -----------------------------
# Time travel: Replay
# -----------------------------
@app.post("/threads/{thread_id}/replay")
def replay(thread_id: str, req: ReplayRequest):
    """
    Replay execution from a prior checkpoint:
    - Uses the checkpoint's config to re-invoke the graph.
    - Nodes after that checkpoint re-execute (LLMs/tools may run again).
    """
    snap = _find_snapshot_by_checkpoint(thread_id, req.checkpoint_id)
    if not snap:
        raise HTTPException(status_code=404, detail="checkpoint_id not found in history")

    # Replay by invoking graph using the checkpoint config.
    # Per LangGraph time-travel docs, invoking with a checkpoint config replays from that point.
    result = GRAPH.invoke(None, snap.config)  # continue from saved state
    return {
        "thread_id": thread_id,
        "replayed_from_checkpoint_id": req.checkpoint_id,
        "answer": result.get("answer", ""),
        "state": result,
    }


# -----------------------------
# Time travel: Fork
# -----------------------------
@app.post("/threads/{thread_id}/fork")
def fork(thread_id: str, req: ForkRequest):
    """
    Fork: create a new thread from a past checkpoint OR latest state.
    The original thread remains unchanged.
    """
    new_thread_id = req.new_thread_id or str(uuid4())

    # Choose base snapshot
    if req.checkpoint_id:
        base = _find_snapshot_by_checkpoint(thread_id, req.checkpoint_id)
        if not base:
            raise HTTPException(status_code=404, detail="checkpoint_id not found in history")
        base_values = base.values
        base_ckpt = req.checkpoint_id
    else:
        latest = GRAPH.get_state(cfg(thread_id))
        base_values = latest.values
        base_ckpt = latest.config["configurable"].get("checkpoint_id")

    # Seed new thread with base state
    if not hasattr(GRAPH, "update_state"):
        raise HTTPException(status_code=500, detail="GRAPH.update_state not available; upgrade langgraph")

    GRAPH.update_state(cfg(new_thread_id), base_values)

    return {
        "source_thread_id": thread_id,
        "forked_thread_id": new_thread_id,
        "forked_from_checkpoint_id": base_ckpt,
        "note": "Use forked_thread_id for subsequent /run calls.",
    }


# -----------------------------
# Time travel: Patch state
# -----------------------------
@app.post("/threads/{thread_id}/patch")
def patch_state(thread_id: str, req: PatchRequest):
    """
    Patch state:
    - If checkpoint_id is provided, the patch branches from that checkpoint.
    - If not provided, patch applies to the latest state in the thread.
    """
    if not hasattr(GRAPH, "update_state"):
        raise HTTPException(status_code=500, detail="GRAPH.update_state not available; upgrade langgraph")

    target_config = cfg(thread_id, req.checkpoint_id) if req.checkpoint_id else cfg(thread_id)

    # Apply update; this creates a new checkpoint (time-travel edit).
    # Optional as_node can be used to attribute the write to a node (advanced).
    if req.as_node:
        GRAPH.update_state(target_config, req.update, as_node=req.as_node)
    else:
        GRAPH.update_state(target_config, req.update)

    # Return new current state
    snap = GRAPH.get_state(cfg(thread_id))
    return {
        "thread_id": thread_id,
        "patched_from_checkpoint_id": req.checkpoint_id,
        "new_checkpoint_id": snap.config["configurable"].get("checkpoint_id"),
        "values": snap.values,
        "next": list(snap.next),
    }