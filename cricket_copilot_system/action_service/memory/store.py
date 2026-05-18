import json
from typing import Any, Dict, List, Optional
from db.postgres import fetch_all, execute

def load_memory(thread_id: str, limit: int = 20) -> Dict[str, Any]:
    cols, rows = fetch_all(
        """
        SELECT role, content, metadata, created_at
        FROM memory.conversation_messages
        WHERE thread_id = %(thread_id)s
        ORDER BY created_at DESC
        LIMIT %(limit)s
        """,
        {"thread_id": thread_id, "limit": limit},
    )
    # Reverse to chronological order
    rows = list(reversed(rows))

    history = []
    for role, content, metadata, _created_at in rows:
        history.append({"role": role, "content": content})

    # Summary
    s_cols, s_rows = fetch_all(
        """
        SELECT summary
        FROM memory.thread_summary
        WHERE thread_id = %(thread_id)s
        """,
        {"thread_id": thread_id},
    )
    summary = s_rows[0][0] if s_rows else ""

    return {"history": history, "summary": summary}

def save_message(thread_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
    execute(
        """
        INSERT INTO memory.conversation_messages (thread_id, role, content, metadata)
        VALUES (%(thread_id)s, %(role)s, %(content)s, %(metadata)s::jsonb)
        """,
        {
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "metadata": json.dumps(metadata or {}),
        },
    )

def upsert_summary(thread_id: str, summary: str):
    execute(
        """
        INSERT INTO memory.thread_summary (thread_id, summary)
        VALUES (%(thread_id)s, %(summary)s)
        ON CONFLICT (thread_id)
        DO UPDATE SET summary = EXCLUDED.summary, updated_at = now()
        """,
        {"thread_id": thread_id, "summary": summary},
    )