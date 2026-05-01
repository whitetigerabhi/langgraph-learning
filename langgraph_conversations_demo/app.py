from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from typing import Optional, Any, Dict, List
import os
import sqlite3
import time

from graph import GRAPH

app = FastAPI(title="LangGraph Conversations Demo", version="1.0")

DB_PATH = os.environ.get("CONV_DB", "./storage/conversations.sqlite")

def db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        user_id TEXT NOT NULL,
        conversation_id TEXT NOT NULL,
        title TEXT NOT NULL,
        current_thread_id TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        PRIMARY KEY (user_id, conversation_id)
    )
    """)
    return conn

def cfg(thread_id: str, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
    c = {"configurable": {"thread_id": thread_id}}
    if checkpoint_id:
        c["configurable"]["checkpoint_id"] = checkpoint_id
    return c

# ---- Models
class NewConversationRequest(BaseModel):
    user_id: str
    title: str

class RenameRequest(BaseModel):
    user_id: str
    title: str

class RunRequest(BaseModel):
    user_id: str
    query: str

class ApproveRequest(BaseModel):
    user_id: str
    approval_token: str

# ---- Conversation CRUD
@app.post("/conversations/new")
def new_conversation(req: NewConversationRequest):
    conversation_id = str(uuid4())
    thread_id = str(uuid4())
    now = int(time.time())

    conn = db()
    conn.execute(
        "INSERT INTO conversations(user_id,conversation_id,title,current_thread_id,created_at,updated_at) VALUES(?,?,?,?,?,?)",
        (req.user_id, conversation_id, req.title, thread_id, now, now)
    )
    conn.commit()
    conn.close()
    return {"conversation_id": conversation_id, "title": req.title, "thread_id": thread_id}

@app.get("/conversations")
def list_conversations(user_id: str):
    conn = db()
    rows = conn.execute(
        "SELECT conversation_id,title,current_thread_id,created_at,updated_at FROM conversations WHERE user_id=? ORDER BY updated_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [{"conversation_id": r[0], "title": r[1], "thread_id": r[2], "created_at": r[3], "updated_at": r[4]} for r in rows]

@app.post("/conversations/{conversation_id}/rename")
def rename_conversation(conversation_id: str, req: RenameRequest):
    now = int(time.time())
    conn = db()
    cur = conn.execute(
        "UPDATE conversations SET title=?, updated_at=? WHERE user_id=? AND conversation_id=?",
        (req.title, now, req.user_id, conversation_id)
    )
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "title": req.title}

def get_thread(user_id: str, conversation_id: str) -> str:
    conn = db()
    row = conn.execute(
        "SELECT current_thread_id FROM conversations WHERE user_id=? AND conversation_id=?",
        (user_id, conversation_id)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return row[0]

def set_thread(user_id: str, conversation_id: str, new_thread_id: str):
    now = int(time.time())
    conn = db()
    cur = conn.execute(
        "UPDATE conversations SET current_thread_id=?, updated_at=? WHERE user_id=? AND conversation_id=?",
        (new_thread_id, now, user_id, conversation_id)
    )
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")

# ---- Run
@app.post("/conversations/{conversation_id}/run")
def run(conversation_id: str, req: RunRequest):
    thread_id = get_thread(req.user_id, conversation_id)
    result = GRAPH.invoke({"query": req.query}, config=cfg(thread_id))

    # If approval required, return a special response
    if result.get("needs_approval") and not result.get("approved"):
        return {
            "conversation_id": conversation_id,
            "thread_id": thread_id,
            "needs_approval": True,
            "approval_token": result.get("approval_token"),
            "proposed": result.get("proposed_answer", ""),
            "state": result,
        }

    return {
        "conversation_id": conversation_id,
        "thread_id": thread_id,
        "needs_approval": False,
        "answer": result.get("answer", ""),
        "state": result,
    }

# ---- State & History
@app.get("/conversations/{conversation_id}/state")
def state(conversation_id: str, user_id: str):
    thread_id = get_thread(user_id, conversation_id)
    snap = GRAPH.get_state(cfg(thread_id))
    return {"conversation_id": conversation_id, "thread_id": thread_id, "values": snap.values, "next": list(snap.next)}

@app.get("/conversations/{conversation_id}/history")
def history(conversation_id: str, user_id: str, limit: int = 20):
    thread_id = get_thread(user_id, conversation_id)
    snaps = []
    for s in GRAPH.get_state_history(cfg(thread_id), limit=limit):
        snaps.append({
            "checkpoint_id": s.config["configurable"].get("checkpoint_id"),
            "next": list(s.next),
            "created_at": getattr(s, "created_at", None),
            "metadata": getattr(s, "metadata", None),
            "values": s.values,
        })
    return {"conversation_id": conversation_id, "thread_id": thread_id, "snapshots": snaps}

# ---- Undo: fork from previous checkpoint and rewire conversation -> new thread
@app.post("/conversations/{conversation_id}/undo")
def undo(conversation_id: str, user_id: str):
    thread_id = get_thread(user_id, conversation_id)
    hist = list(GRAPH.get_state_history(cfg(thread_id), limit=5))
    if len(hist) < 2:
        raise HTTPException(status_code=400, detail="Not enough history to undo")

    # Pick the previous snapshot (index 1, because history is newest-first)
    prev = hist[1]
    prev_values = prev.values

    new_thread_id = str(uuid4())

    # Time travel fork by seeding new thread with old values
    try:
        GRAPH.update_state(cfg(new_thread_id), prev_values)  # forks via checkpointing semantics [4](https://www.thedataschool.co.uk/a/chris-meardon/an-api-to-remember-postcodes-io/)[6](https://postcodes.io/endpoints/)
    except AttributeError:
        raise HTTPException(status_code=500, detail="update_state not available; upgrade langgraph")

    # Rewire conversation to new thread (user sees same conversation)
    set_thread(user_id, conversation_id, new_thread_id)

    return {"conversation_id": conversation_id, "old_thread_id": thread_id, "new_thread_id": new_thread_id, "undone_to_checkpoint": prev.config["configurable"].get("checkpoint_id")}

# ---- Redo: replay from latest checkpoint (simple approach: just rerun last query)
@app.post("/conversations/{conversation_id}/redo")
def redo(conversation_id: str, user_id: str):
    thread_id = get_thread(user_id, conversation_id)
    snap = GRAPH.get_state(cfg(thread_id))
    last_query = (snap.values or {}).get("query") or (snap.values or {}).get("normalized_query")
    if not last_query:
        raise HTTPException(status_code=400, detail="No prior query to redo")

    result = GRAPH.invoke({"query": last_query}, config=cfg(thread_id))
    return {"conversation_id": conversation_id, "thread_id": thread_id, "answer": result.get("answer", ""), "state": result}

# ---- Approval: mark approved and finalize
@app.post("/conversations/{conversation_id}/approve")
def approve(conversation_id: str, req: ApproveRequest):
    thread_id = get_thread(req.user_id, conversation_id)
    snap = GRAPH.get_state(cfg(thread_id))
    values = snap.values or {}

    if values.get("approval_token") != req.approval_token:
        raise HTTPException(status_code=400, detail="Invalid approval_token")

    # Mark approved in state and re-run to finalize
    try:
        GRAPH.update_state(cfg(thread_id), {"approved": True})  # apply partial state update [6](https://postcodes.io/endpoints/)[4](https://www.thedataschool.co.uk/a/chris-meardon/an-api-to-remember-postcodes-io/)
    except AttributeError:
        raise HTTPException(status_code=500, detail="update_state not available; upgrade langgraph")

    result = GRAPH.invoke({"query": values.get("query", "")}, config=cfg(thread_id))
    return {"conversation_id": conversation_id, "thread_id": thread_id, "approved": True, "answer": result.get("answer", ""), "state": result}